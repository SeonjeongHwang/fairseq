# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
    RearTruncateDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from examples.roberta.multiprocessing_bpe_encoder import *
from examples.roberta.preprocess_CoQA import *


logger = logging.getLogger(__name__)


@register_task("CoQA")
class CoQATask(LegacyFairseqTask):
    """
    Ranking task on multiple sentences.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument("--encoder-json", help="path to encoder.json")
        parser.add_argument("--vocab-bpe", type=str, help="path to vocab.bpe")
        parser.add_argument(
            "--init-token",
            type=int,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token", type=int, help="add separator token between inputs"
        )
        parser.add_argument("--no-shuffle", action="store_true")
        parser.add_argument(
            "--max-query-length", type=int, help="max length of each query (histories+query)"
        )

        parser.add_argument(
            "--max-seq-length", type=int, help="max length of each input sequence"
        )
        parser.add_argument(
            "--doc-stride", type=int, help=""
        )
        parser.add_argument(
            "--num-turns", type=int, help="number of history turns"
        )
        
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        cls.Q_token = dictionary.add_symbol("<Q>")
        cls.A_token = dictionary.add_symbol("<A>")
        
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert (
            args.criterion == "sentence_ranking"
        ), "Must set --criterion=sentence_ranking"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            "dict.txt",
            source=True,
        )
        logger.info("Dictionary: {} types".format(len(data_dict)))
        return CoQATask(args, data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        
        ###encoder 객체 생성
        bpe_encoder = MultiprocessingEncoder(self.args.encoder_json, self.args.vocab_bpe)
        bpe_encoder.initializer()
        
        ###preprocess_coqa부르기
        features = get_CoQA_features(self.args, bpe_encoder, split=="train")
        
        src_tokens = []
        src_lengths = []
        start_pos = []
        end_pos = []
        is_unk = []
        is_yes = []
        is_no = []
        number = []
        option = []
        
        for feature in features:
            #history들과 query 이어붙이고, max_query_length로 자르기(RearTruncate)
            query = []
            for q, a in zip(feature.history_q, feature.history_a):
                query.append(self.Dictionary.index("<Q>"))
                query.extend(q)
                query.append(self.Dictionary.index("<A>"))
                query.extend(a)
            query.append(self.Dictionary.index("<Q>"))
            query.extend(feature.query)
                            
            query = np.array(query)
            query = torch.IntTensor(query).long()
            
            query = RearTruncateDataset(
                    input_option, self.args.max_query_length
                )
            
            ##[CLS]
            src = PrependTokenDataset(query, self.args.init_token)
            
            context = np.array(feature.para)
            context = torch.IntTensor(context).long()
            
            ##[SEP]
            context = PrependTokenDataset(context, self.args.seperator_token)
            
            ##+context
            src_token = ConcatSentencesDataset(context, src)
            ##last [SEP]
            src = AppendTokenDataset(src, self.args.seperator_token)
            
            src_length = len(src)
            
            src_tokens.append(src)
            src_lengths.append(src_length)
            
            start_pos = feature.start_position
            end_pos = feature.end_position
            is_unk = feature.is_unk
            is_yes = feature.is_yes
            is_no = feature.is_no
            number = feature.number
            option = feature.option
        
        src_lengths = np.array(src_lengths)
        src_tokens = ListDataset(src_tokens, src_lengths)
        src_lenfths = ListDataset(src_lengths)
        
        dataset = {
            "id": IdDataset(),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.Dictionary.pad()),
            "src_lengths": RawLabelDataset(src_lengths),
            "start_position": RawLabelDataset(start_pos),
            "end_position": RawLabelDataset(end_pos),
            "is_unk": RawLabelDataset(is_unk),
            "is_yes": RawLabelDataset(is_yes),
            "is_no": RawLabelDataset(is_no),
            "number": RawLabelDataset(number),
            "option": RawLabelDataset(option),     
        }

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )
        
        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                sort_order[np.random.permutation(len(dataset))],
            )
            
        print("| Loaded {} with {} samples".format(split, len(dataset)))
        
        self.datasets[split] = dataset
        return self.datasets[split]
                             
         
    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self) #RobertaEncoder

        model.register_classification_head(
            getattr(args, "ranking_head_name", "sentence_classification_head"),
            num_classes=1,
        )

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
