from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import collections
import os
import os.path
import json
import pickle
import time
import string

import numpy as np

from examples.roberta.tool.eval_coqa import CoQAEvaluator
from examples.roberta.multiprocessing_bpe_encoder import *
import logging

MAX_FLOAT = 1e30
MIN_FLOAT = -1e30

logger = logging.getLogger(__name__)


def get_CoQA_features(args, encoder, cls_idx, sep_idx, pad_idx, split="train"):
    
    task_name = "coqa"
    bpe_encoder = encoder
    
    data_pipeline = CoqaPipeline(
        data_dir=args.data,
        task_name=task_name,
        num_turn=args.num_turns)
    
    example_processor = CoQAExampleProcessor(
        max_seq_length=args.max_positions,
        max_query_length=args.max_query_length,
        doc_stride=args.doc_stride,
        cls_token=cls_idx,
        sep_token=sep_idx,
        pad_token=pad_idx,
        encoder=bpe_encoder)
    
    if split=="train":
        train_examples = data_pipeline.get_train_examples()
        train_features = example_processor.convert_examples_to_features(train_examples)
        return train_examples, train_features
        
    elif split=="valid":
        predict_examples = data_pipeline.get_dev_examples()
        predict_features = example_processor.convert_examples_to_features(predict_examples)
        return predict_examples, predict_features
    
def get_best_predictions(args, examples, features, bpe_encoder, mode="train"):
    """
    <examples>
    example = InputExample(
    qas_id=qas_id,
    question_text=question_text,
    paragraph_text=paragraph_text,
    orig_answer_text=orig_answer_text,
    start_position=start_position,
    answer_type=answer_type,
    answer_subtype=answer_subtype,
    is_skipped=is_skipped)
    
    <features>
    feature = InputFeatures(
    unique_id=self.unique_id,
    qas_id=example.qas_id,
    doc_idx=doc_idx,
    token2char_raw_start_index=doc_token2char_raw_start_index,
    token2char_raw_end_index=doc_token2char_raw_end_index,
    token2doc_index=doc_token2doc_index,
    input_tokens=input_tokens,
    para_start_index=para_start_index,
    para_end_index=para_end_index,
    p_mask=p_mask,
    para_length=doc_para_length,
    start_position=start_position,
    end_position=end_position,
    is_unk=is_unk,
    is_yes=is_yes,
    is_no=is_no,
    number=number,
    option=option)
    
    <preds> => <results>
    pred["unique_id"] = sample["id"].tolist()[i]
    pred["qas_id"] = sample["qas_id"].tolist()[i]
    pred["start_prob"] = preds["start_prob"].tolist()[i]
    pred["start_index"] = preds["start_index"].tolist()[i]
    pred["end_prob"] = preds["end_prob"].tolist()[i]
    pred["end_index"] = preds["end_index"].tolist()[i]
    pred["unk_prob"] = preds["unk_prob"].tolist()[i]
    pred["yes_prob"] = preds["yes_prob"].tolist()[i]
    pred["no_prob"] = preds["no_prob"].tolist()[i]
    pred["num_probs"] = preds["num_probs"].tolist()[i]
    pred["opt_probs"] = preds["opt_probs"].tolist()[i]
    """
    
    def write_to_json(data_list, data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:  
            json.dump(data_list, file, indent=4)
    
    output_summary = os.path.join(args.task.data, "predict.summary.json")
    output_detail = os.path.join(args.task.data, "predict.detail.json")
    
    if mode=="train":
        qas_map_path = os.path.join(args.task.data, "train-qas_id_map.json")
    else:
        qas_map_path = os.path.join(args.task.data, "dev-qas_id_map.json")
        
    qas_id_map = None
    with open(qas_map_path, "r") as f:
        qas_id_map = json.load(f)
    
    prediction_file = args.task.save_predictions
    preds = []
    with open(prediction_file, "r") as f:
        pred_lines = f.readlines()
        for line in pred_lines:
            preds.append(json.loads(line))
    
    qas_id_to_features = {}
    unique_id_to_feature = {}
    for feature in features:
        if feature.qas_id not in qas_id_to_features:
            qas_id_to_features[feature.qas_id] = []

        qas_id_to_features[feature.qas_id].append(feature)
        unique_id_to_feature[feature.unique_id] = feature
        
    unique_id_to_result = {}
    for pred in preds:
        unique_id_to_result[pred["unique_id"]] = pred

    predict_summary_list = []
    predict_detail_list = []
    num_example = len(examples)
    for (example_idx, example) in enumerate(examples):
        if example_idx % 1000 == 0:
            logger.info('Updating {0}/{1} example with predict'.format(example_idx, num_example))

        if example.qas_id not in qas_id_to_features:
            logger.info('No feature found for example: {0}'.format(example.qas_id))
            continue

        example_unk_score = MAX_FLOAT
        example_yes_score = MIN_FLOAT
        example_no_score = MIN_FLOAT
        example_num_id = 0
        example_num_score = MIN_FLOAT
        example_num_probs = None
        example_opt_id = 0
        example_opt_score = MIN_FLOAT
        example_opt_probs = None

        example_all_predicts = []
        example_features = qas_id_to_features[example.qas_id]
        for example_feature in example_features:
            if example_feature.unique_id not in unique_id_to_result:
                logger.info('No result found for feature: {0}'.format(example_feature.unique_id))
                continue

            example_result = unique_id_to_result[example_feature.unique_id]
            example_unk_score = min(example_unk_score, float(example_result["unk_prob"]))
            example_yes_score = max(example_yes_score, float(example_result["yes_prob"]))
            example_no_score = max(example_no_score, float(example_result["no_prob"]))

            num_probs = [float(num_prob) for num_prob in example_result["num_probs"]]
            num_id = int(np.argmax(num_probs[1:])) + 1
            num_score = num_probs[num_id]
            if example_num_score < num_score:
                example_num_id = num_id
                example_num_score = num_score
                example_num_probs = num_probs

            opt_probs = [float(opt_prob) for opt_prob in example_result["opt_probs"]]
            opt_id = int(np.argmax(opt_probs[1:])) + 1
            opt_score = opt_probs[opt_id]
            if example_opt_score < opt_score:
                example_opt_id = opt_id
                example_opt_score = opt_score
                example_opt_probs = opt_probs

                
            for i in range(args.task.start_n_top):
                start_prob = example_result["start_prob"][i]
                start_index = example_result["start_index"][i]

                for j in range(args.task.end_n_top):
                    end_prob = example_result["end_prob"][i][j]
                    end_index = example_result["end_index"][i][j] #input_tokens 상에서의 prediction
                    
                    ##start 보다 end index의 값이 같거나 큰가?
                    if start_index > end_index:
                        continue
                        
                    ##input_tokens 상의 context 범위 내에 있는가?
                    if start_index < example_feature.para_start_index or end_index > example_feature.para_end_index:
                        continue
                        
                    """
                    
                    ##
                    predict_tokens = example_feature.input_tokens[start_index:end_index+1]
                    predict_text = 
                    
                    doc_start_token_idx = start_index - example_feature.para_start_index
                    if doc_start_token_idx < 0: continue
                    raw_start_token_idx = doc_start_token_idx + example_feature.doc_start
                    doc_end_token_idx = end_index - example_feature.para_start_index
                    if doc_end_token_idx < 0: continue
                    raw_end_token_idx = doc_end_token_idx + example_feature.doc_start
                    
                    if raw_start_token_idx > raw_end_token_idx:
                        continue
                        
                    if raw_end_token_idx >= len(example_feature.sis_tokens_index):
                        continue
                    
                    raw_modified_start_token_idx = example_feature.sis_tokens_index[raw_start_token_idx][0]
                    doc_start_index = raw_modified_start_token_idx - example_feature.doc_start
                    raw_modified_end_token_idx = example_feature.sis_tokens_index[raw_end_token_idx][-1]
                    doc_end_index = raw_modified_start_token_idx - example_feature.doc_start
                    
                    answer_length = end_index - start_index + 1
                    if answer_length > args.task.max_answer_length:
                        continue

                    if doc_end_index >= len(example_feature.token2char_raw_start_index):
                        continue
                        
                    if start_index not in example_feature.token2doc_index:
                        continue
                        
                    """    

                    example_all_predicts.append({
                        "unique_id": example_result["unique_id"],
                        "start_prob": float(start_prob),
                        "start_index": int(start_index),
                        "end_prob": float(end_prob),
                        "end_index": int(end_index),
                        "predict_score": float(np.log(start_prob) + np.log(end_prob))
                    })
                    
        example_all_predicts = sorted(example_all_predicts, key=lambda x: x["predict_score"], reverse=True)

        is_visited = set()
        example_top_predicts = []
        for example_predict in example_all_predicts:
            if len(example_top_predicts) >= args.criterion.n_best_size:
                break
            
            predict_tokens = example_feature.input_tokens[start_index:end_index+1]
            predict_text = bpe_encoder.decode_tokens(predict_tokens)
            
            """
            predict_start = example_feature.token2char_raw_start_index[doc_start_index]
            predict_end = example_feature.token2char_raw_end_index[doc_end_index]
            predict_text = example.paragraph_text[predict_start:predict_end + 1].strip()
            """

            if predict_text in is_visited:
                continue

            is_visited.add(predict_text)

            example_top_predicts.append({
                "predict_text": predict_text,
                "predict_score": example_predict["predict_score"]
            })

        if len(example_top_predicts) == 0:
            example_top_predicts.append({
                "predict_text": "",
                "predict_score": 0.0
            })

        example_best_predict = example_top_predicts[0]

        example_question_text = example.question_text.split('<s>')[-1].strip()

        predict_summary_list.append({
            "befor_id": example.qas_id,
            "qas_id": qas_id_map[str(example.qas_id)],
            "question_text": example_question_text,
            "label_text": example.orig_answer_text,
            "unk_score": example_unk_score,
            "yes_score": example_yes_score,
            "no_score": example_no_score,
            "num_id": example_num_id,
            "num_score": example_num_score,
            "num_probs": example_num_probs,
            "opt_id": example_opt_id,
            "opt_score": example_opt_score,
            "opt_probs": example_opt_probs,
            
            "predict_text": example_best_predict["predict_text"],
            "predict_score": example_best_predict["predict_score"]
        })

        predict_detail_list.append({
            "qas_id": qas_id_map[str(example.qas_id)],
            "question_text": example_question_text,
            "label_text": example.orig_answer_text,
            "unk_score": example_unk_score,
            "yes_score": example_yes_score,
            "no_score": example_no_score,
            "num_id": example_num_id,
            "num_score": example_num_score,
            "num_probs": example_num_probs,
            "opt_id": example_opt_id,
            "opt_score": example_opt_score,
            "opt_probs": example_opt_probs,
            "best_predict": example_best_predict,
            "top_predicts": example_top_predicts
        })
        
    write_to_json(predict_summary_list, output_summary)
    write_to_json(predict_detail_list, output_detail)       
        
    return predict_summary_list
         

class InputExample(object):
    """A single CoQA example."""
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 answer_type=None,
                 answer_subtype=None,
                 is_skipped=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.answer_type = answer_type
        self.answer_subtype = answer_subtype
        self.is_skipped = is_skipped
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = "qas_id: %s" % (prepro_utils.printable_text(self.qas_id))
        s += ", question_text: %s" % (prepro_utils.printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (prepro_utils.printable_text(self.paragraph_text))
        if self.start_position >= 0:
            s += ", start_position: %d" % (self.start_position)
            s += ", orig_answer_text: %s" % (prepro_utils.printable_text(self.orig_answer_text))
            s += ", answer_type: %s" % (prepro_utils.printable_text(self.answer_type))
            s += ", answer_subtype: %s" % (prepro_utils.printable_text(self.answer_subtype))
            s += ", is_skipped: %r" % (self.is_skipped)
        return "[{0}]\n".format(s)

class InputFeatures(object):
    """A single CoQA feature."""
    def __init__(self,
                 unique_id,
                 qas_id,
                 doc_idx,
                 doc_start,
                 sis_tokens_index,
                 token2char_raw_start_index,
                 token2char_raw_end_index,
                 token2doc_index,
                 input_tokens,
                 src_length,
                 para_start_index,
                 para_end_index,
                 p_mask,
                 para_length,
                 start_position=None,
                 end_position=None,
                 is_unk=None,
                 is_yes=None,
                 is_no=None,
                 number=None,
                 option=None):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.doc_idx = doc_idx
        self.doc_start = doc_start
        self.sis_tokens_index = sis_tokens_index
        self.token2char_raw_start_index = token2char_raw_start_index
        self.token2char_raw_end_index = token2char_raw_end_index
        self.token2doc_index = token2doc_index
        self.input_tokens = input_tokens
        self.src_length = src_length
        self.para_start_index = para_start_index
        self.para_end_index = para_end_index
        self.p_mask = p_mask
        self.para_length = para_length
        self.start_position = start_position
        self.end_position = end_position
        self.is_unk = is_unk
        self.is_yes = is_yes
        self.is_no = is_no
        self.number = number
        self.option = option

###1
class CoqaPipeline(object):
    """Pipeline for CoQA dataset."""
    def __init__(self,
                 data_dir,
                 task_name,
                 num_turn):
        self.data_dir = data_dir
        self.task_name = task_name
        self.num_turn = num_turn
        self.train_mode = True
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        example_list = [example for example in example_list if not example.is_skipped]
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        self.train_mode = False
        data_path = os.path.join(self.data_dir, "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)["data"]
                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(data_path))
    
    def _whitespace_tokenize(self,
                             text):
        word_spans = []
        char_list = []
        for idx, char in enumerate(text):
            if char != ' ':
                char_list.append(idx)
                continue
            
            if char_list:
                word_start = char_list[0]
                word_end = char_list[-1]
                word_text = text[word_start:word_end+1]
                word_spans.append((word_text, word_start, word_end))
                char_list.clear()
        
        if char_list:
            word_start = char_list[0]
            word_end = char_list[-1]
            word_text = text[word_start:word_end+1]
            word_spans.append((word_text, word_start, word_end))
        
        return word_spans
    
    def _char_span_to_word_span(self,
                                char_start,
                                char_end,
                                word_spans):
        word_idx_list = []
        for word_idx, (_, start, end) in enumerate(word_spans):
            if end >= char_start:
                if start <= char_end:
                    word_idx_list.append(word_idx)
                else:
                    break
        
        if word_idx_list:
            word_start = word_idx_list[0]
            word_end = word_idx_list[-1]
        else:
            word_start = -1
            word_end = -1
        
        return word_start, word_end
    
    def _search_best_span(self,
                          context_tokens,
                          answer_tokens):
        best_f1 = 0.0
        best_start, best_end = -1, -1
        search_index = [idx for idx in range(len(context_tokens)) if context_tokens[idx][0] in answer_tokens]
        for i in range(len(search_index)):
            for j in range(i, len(search_index)):
                candidate_tokens = [context_tokens[k][0] for k in range(search_index[i], search_index[j]+1) if context_tokens[k][0]]
                common = collections.Counter(candidate_tokens) & collections.Counter(answer_tokens)
                num_common = sum(common.values())
                if num_common > 0:
                    precision = 1.0 * num_common / len(candidate_tokens)
                    recall = 1.0 * num_common / len(answer_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_start = context_tokens[search_index[i]][1]
                        best_end = context_tokens[search_index[j]][2]
        
        return best_f1, best_start, best_end
    
    def _get_question_text(self,
                           history,
                           question):
        question_tokens = ['<s>'] + question["input_text"].split(' ')
        return " ".join(history + [" ".join(question_tokens)])
    
    def _get_question_history(self,
                              history,
                              question,
                              answer,
                              answer_type,
                              is_skipped,
                              num_turn):
        question_tokens = []
        if answer_type != "unknown":
            question_tokens.extend(['<s>'] + question["input_text"].split(' '))
            question_tokens.extend(['</s>'] + answer["input_text"].split(' '))
        
        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)
        
        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]
        
        return history
    
    def _find_answer_span(self,
                          answer_text,
                          rationale_text,
                          rationale_start,
                          rationale_end):
        idx = rationale_text.find(answer_text)
        answer_start = rationale_start + idx
        answer_end = answer_start + len(answer_text) - 1
        
        return answer_start, answer_end
    
    def _match_answer_span(self,
                           answer_text,
                           rationale_start,
                           rationale_end,
                           paragraph_text):
        answer_tokens = self._whitespace_tokenize(answer_text)
        answer_norm_tokens = [CoQAEvaluator.normalize_answer(token) for token, _, _ in answer_tokens]
        answer_norm_tokens = [norm_token for norm_token in answer_norm_tokens if norm_token]
        
        if not answer_norm_tokens:
            return -1, -1
        
        paragraph_tokens = self._whitespace_tokenize(paragraph_text)
        
        if not (rationale_start == -1 or rationale_end == -1):
            rationale_word_start, rationale_word_end = self._char_span_to_word_span(rationale_start, rationale_end, paragraph_tokens)
            rationale_tokens = paragraph_tokens[rationale_word_start:rationale_word_end+1]
            rationale_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in rationale_tokens]
            match_score, answer_start, answer_end = self._search_best_span(rationale_norm_tokens, answer_norm_tokens)
            
            if match_score > 0.0:
                return answer_start, answer_end
        
        paragraph_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in paragraph_tokens]
        match_score, answer_start, answer_end = self._search_best_span(paragraph_norm_tokens, answer_norm_tokens)
        
        if match_score > 0.0:
            return answer_start, answer_end
        
        return -1, -1
    
    def _get_answer_span(self,
                         answer,
                         answer_type,
                         paragraph_text):
        input_text = answer["input_text"].strip().lower()
        span_start, span_end = answer["span_start"], answer["span_end"]
        if span_start == -1 or span_end == -1:
            span_text = ""
        else:
            span_text = paragraph_text[span_start:span_end].lower()
        
        if input_text in span_text:
            span_start, span_end = self._find_answer_span(input_text, span_text, span_start, span_end)
        else:
            span_start, span_end = self._match_answer_span(input_text, span_start, span_end, paragraph_text.lower())
        
        if span_start == -1 or span_end == -1:
            answer_text = ""
            is_skipped = (answer_type == "span")
        else:
            answer_text = paragraph_text[span_start:span_end+1]
            is_skipped = False
        
        return answer_text, span_start, span_end, is_skipped
    
    def _normalize_answer(self,
                          answer):
        norm_answer = CoQAEvaluator.normalize_answer(answer)
        
        if norm_answer in ["yes", "yese", "ye", "es"]:
            return "yes"
        
        if norm_answer in ["no", "no not at all", "not", "not at all", "not yet", "not really"]:
            return "no"
        
        return norm_answer
    
    def _get_answer_type(self,
                         question,
                         answer):
        norm_answer = self._normalize_answer(answer["input_text"])
        
        if norm_answer == "unknown" or "bad_turn" in answer:
            return "unknown", None
        
        if norm_answer == "yes":
            return "yes", None
        
        if norm_answer == "no":
            return "no", None
        
        if norm_answer in ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            return "number", norm_answer
        
        norm_question_tokens = CoQAEvaluator.normalize_answer(question["input_text"]).split(" ")
        if "or" in norm_question_tokens:
            index = norm_question_tokens.index("or")
            if index-1 >= 0 and index+1 < len(norm_question_tokens):
                if norm_answer == norm_question_tokens[index-1]:
                    norm_answer = "option_a"
                elif norm_answer == norm_question_tokens[index+1]:
                    norm_answer = "option_b"
        
        if norm_answer in ["option_a", "option_b"]:
            return "option", norm_answer
        
        return "span", None
    
    def _process_found_answer(self,
                              raw_answer,
                              found_answer):
        raw_answer_tokens = raw_answer.split(' ')
        found_answer_tokens = found_answer.split(' ')
        
        raw_answer_last_token = raw_answer_tokens[-1].lower()
        found_answer_last_token = found_answer_tokens[-1].lower()
        
        if (raw_answer_last_token != found_answer_last_token and
            raw_answer_last_token == found_answer_last_token.rstrip(string.punctuation)):
            found_answer_tokens[-1] = found_answer_tokens[-1].rstrip(string.punctuation)
        
        return ' '.join(found_answer_tokens)
    
    def _get_example(self,
                     data_list):
        if self.train_mode:
            qas_map_path = os.path.join(self.data_dir, "train-qas_id_map.json")
        else:
            qas_map_path = os.path.join(self.data_dir, "dev-qas_id_map.json")
        
        examples = []
        qas_map = {}
        qas_new_id = 0
        for data in data_list:
            data_id = data["id"]
            paragraph_text = data["story"]
            
            questions = sorted(data["questions"], key=lambda x: x["turn_id"])
            answers = sorted(data["answers"], key=lambda x: x["turn_id"])
            
            question_history = []
            qas = list(zip(questions, answers))
            for i, (question, answer) in enumerate(qas):
                qas_id = "{0}_{1}".format(data_id, i+1)
                qas_map[qas_new_id] = qas_id
                qas_id = qas_new_id
                qas_new_id += 1
                
                answer_type, answer_subtype = self._get_answer_type(question, answer)
                answer_text, span_start, span_end, is_skipped = self._get_answer_span(answer, answer_type, paragraph_text)
                question_text = self._get_question_text(question_history, question)
                question_history = self._get_question_history(question_history, question, answer, answer_type, is_skipped, self.num_turn)
                
                if answer_type not in ["unknown", "yes", "no"] and not is_skipped and answer_text:
                    start_position = span_start
                    orig_answer_text = self._process_found_answer(answer["input_text"], answer_text)
                else:
                    start_position = -1
                    orig_answer_text = ""
                
                example = InputExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    answer_type=answer_type,
                    answer_subtype=answer_subtype,
                    is_skipped=is_skipped)

                examples.append(example)
                
        with open(qas_map_path, "w") as o:
            json.dump(qas_map, o)
        
        return examples


class CoQAExampleProcessor(object):
    """Default example processor for CoQA"""
    def __init__(self,
                 max_seq_length,
                 max_query_length,
                 doc_stride,
                 cls_token,
                 sep_token,
                 pad_token,
                 encoder):
        
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.bpe_encoder = encoder
        self.unique_id = 0
        self.f = open("data/coqa/example_check.txt", "w")
    
    def _find_max_context(self,
                          doc_spans,
                          token_idx):
        """Check if this is the 'max context' doc span for the token.

        Because of the sliding window approach taken to scoring documents, a single
        token can appear in multiple documents. E.g.
          Doc: the man went to the store and bought a gallon of milk
          Span A: the man went to the
          Span B: to the store and bought
          Span C: and bought a gallon of
          ...
        
        Now the word 'bought' will have two scores from spans B and C. We only
        want to consider the score with "maximum context", which we define as
        the *minimum* of its left and right context (the *sum* of left and
        right context will always be the same, of course).
        
        In the example the maximum context for 'bought' would be span C since
        it has 1 left context and 3 right context, while span B has 4 left context
        and 0 right context.
        """
        best_doc_score = None
        best_doc_idx = None
        for (doc_idx, doc_span) in enumerate(doc_spans):
            doc_start = doc_span["start"]
            doc_length = doc_span["length"]
            doc_end = doc_start + doc_length - 1
            if token_idx < doc_start or token_idx > doc_end:
                continue
            
            left_context_length = token_idx - doc_start
            right_context_length = doc_end - token_idx
            doc_score = min(left_context_length, right_context_length) + 0.01 * doc_length
            if best_doc_score is None or doc_score > best_doc_score:
                best_doc_score = doc_score
                best_doc_idx = doc_idx
        
        return best_doc_idx
    
    ###
    def convert_coqa_example(self,
                             example,
                             logging=False):
        """Converts a single `InputExample` into a single `InputFeatures`.     
        """
        
        query_tokens = []
        qa_texts = example.question_text.split('<s>')
        for qa_text in qa_texts:
            qa_text = qa_text.strip()
            if not qa_text:
                continue
            
            qa_items = qa_text.split('</s>')
            if len(qa_items) < 1:
                continue
            
            q_text = qa_items[0].strip()
            q_tokens, _, _, _, _ = self.bpe_encoder.encode_line("Q: " + q_text) #bpe_tokens, char2token, token2startchar, token2endchar
            query_tokens.extend(q_tokens)
            
            if len(qa_items) < 2:
                self.f.write("\nQuery:"+q_text+"\n")
                continue
            
            a_text = qa_items[1].strip()
            a_tokens, _, _, _, _ = self.bpe_encoder.encode_line("A: " + a_text)
            query_tokens.extend(a_tokens)
            
        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[-self.max_query_length:]
        
        para_text = example.paragraph_text
        para_tokens, char2token_index, sis_tokens_index, token2char_raw_start_index, token2char_raw_end_index = self.bpe_encoder.encode_line(para_text)
        
        self.f.write("Original answer:"+example.orig_answer_text+"\n")
        answer_start_index = example.start_position
        answer_end_index = example.start_position + len(example.orig_answer_text) - 1
        self.f.write("Using index:"+para_text[answer_start_index:answer_end_index]+"\n")
        answer_start_token = char2token_index[answer_start_index]
        answer_start_token = sis_tokens_index[answer_start_token][0]
        answer_end_token = char2token_index[answer_end_index]
        answer_end_token = sis_tokens_index[answer_end_token][-1]
        answer_tokens = para_tokens[answer_start_token:answer_end_token+1] #### +1?
        self.f.write("After BPE:"+self.bpe_encoder.decode_tokens(answer_tokens)+"\n")
        
        if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
            raw_start_char_pos = example.start_position
            raw_end_char_pos = raw_start_char_pos + len(example.orig_answer_text) - 1
            tokenized_start_token_pos = char2token_index[raw_start_char_pos]
            tokenized_start_token_pos = sis_tokens_index[tokenized_start_token_pos][0]
            tokenized_end_token_pos = char2token_index[raw_end_char_pos]
            tokenized_end_token_pos = sis_tokens_index[tokenized_end_token_pos][-1]
            assert tokenized_start_token_pos <= tokenized_end_token_pos
        else:
            tokenized_start_token_pos = tokenized_end_token_pos = -1
        
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_para_length = self.max_seq_length - len(query_tokens) - 3
        total_para_length = len(para_tokens)
        
        #################################################################################################
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        doc_spans = []
        para_start = 0
        while para_start < total_para_length:
            para_length = total_para_length - para_start
            if para_length > max_para_length:
                para_length = max_para_length
            
            doc_spans.append({
                "start": para_start,
                "length": para_length
            })
            
            if para_start + para_length == total_para_length:
                break
            
            para_start += min(para_length, self.doc_stride)
        
        feature_list = []
        for (doc_idx, doc_span) in enumerate(doc_spans):
            input_tokens = []
            p_mask = []
            doc_token2char_raw_start_index = []
            doc_token2char_raw_end_index = []
            doc_sis_tokens_index = []
            doc_token2doc_index = {} #이 토큰이 몇번째 document에 속하는지??
            
            input_tokens.append(self.cls_token)
            
            for query_token in query_tokens:
                input_tokens.append(query_token)
            
            input_tokens.append(self.sep_token)
            para_start_index = len(input_tokens)
            
            doc_tokens = []
            for i in range(doc_span["length"]): #token 단위로 처리
                token_idx = doc_span["start"] + i
                
                doc_token2char_raw_start_index.append(token2char_raw_start_index[token_idx])
                doc_token2char_raw_end_index.append(token2char_raw_end_index[token_idx])
                doc_sis_tokens_index.append(sis_tokens_index[token_idx])
                
                best_doc_idx = self._find_max_context(doc_spans, token_idx)
                doc_token2doc_index[len(doc_tokens)] = (best_doc_idx == doc_idx)
                
                doc_tokens.append(para_tokens[token_idx])
                input_tokens.append(para_tokens[token_idx])
            
            para_end_index = len(input_tokens)-1
                
            input_tokens.append(self.sep_token)
            p_mask = [0] * len(input_tokens)
            src_length = len(input_tokens)
            
            for _ in range(self.max_seq_length - len(input_tokens)):
                input_tokens.append(self.pad_token)
                p_mask.append(1)
                
            assert len(input_tokens)==self.max_seq_length
            assert len(p_mask)==self.max_seq_length
                
            
            doc_para_length = len(doc_tokens)
            cls_index = 0
            
            start_position = None
            end_position = None
            is_unk = 1 if (example.answer_type == "unknown" or example.is_skipped) else 0
            is_yes = 1 if example.answer_type == "yes" else 0
            is_no = 1 if example.answer_type == "no" else 0
            
            if example.answer_type == "number":
                number_list = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
                number = number_list.index(example.answer_subtype) + 1
            else:
                number = 0
            
            if example.answer_type == "option":
                option_list = ["option_a", "option_b"]
                option = option_list.index(example.answer_subtype) + 1
            else:
                option = 0
            
            doc_start = doc_span["start"]
            doc_end = doc_start + doc_span["length"] - 1
            if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
                if tokenized_start_token_pos >= doc_start and tokenized_end_token_pos <= doc_end:                    
                    start_position = tokenized_start_token_pos - doc_start + para_start_index #tokenized_start_token_pos : paragraph에서의 pos
                    end_position = tokenized_end_token_pos - doc_start + para_start_index
                    
                    answer_tokens = input_tokens[start_position:end_position+1]
                    final_answer_text = self.bpe_encoder.decode_tokens(answer_tokens)
                    self.f.write("After split:"+final_answer_text+"\n\n")
                else:
                    start_position = cls_index
                    end_position = cls_index
                    is_unk = True
            else:
                start_position = cls_index
                end_position = cls_index
               
            feature = InputFeatures(
                unique_id=self.unique_id,
                qas_id=example.qas_id,
                doc_idx=doc_idx,
                doc_start=doc_start,
                sis_tokens_index=doc_sis_tokens_index,
                token2char_raw_start_index=doc_token2char_raw_start_index,
                token2char_raw_end_index=doc_token2char_raw_end_index,
                token2doc_index=doc_token2doc_index,
                input_tokens=input_tokens,
                src_length=src_length,
                para_start_index=para_start_index,
                para_end_index=para_end_index,
                p_mask=p_mask,
                para_length=doc_para_length,
                start_position=start_position,
                end_position=end_position,
                is_unk=is_unk,
                is_yes=is_yes,
                is_no=is_no,
                number=number,
                option=option)
            
            
            feature_list.append(feature)
            self.unique_id += 1
        
        return feature_list
    
    def convert_examples_to_features(self,
                                     examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in enumerate(examples):
            if idx % 1000 == 0:
                print("Converting example %d of %d" % (idx, len(examples)))

            feature_list = self.convert_coqa_example(example, logging=(idx < 20))
            features.extend(feature_list)
        
        print("Generate %d features from %d examples" % (len(features), len(examples)))
        
        return features
        