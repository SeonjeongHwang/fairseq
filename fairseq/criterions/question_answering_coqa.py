# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import json

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

MAX_FLOAT = 1e30
MIN_FLOAT = -1e30


@register_criterion("coqa")
class CoqaCriterion(FairseqCriterion):
    def __init__(self, task, ranking_head_name, save_predictions):
        super().__init__(task)
        self.ranking_head_name = ranking_head_name
        self.start_n_top = 5 ##################################
        self.end_n_top = 5
        if save_predictions is not None:
            self.prediction_h = True
        else:
            self.prediction_h = None

    def __del__(self):
        pass
        #if self.prediction_h is not None:
        #    self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--ranking-head-name',
                            default='coqa',
                            help='name of the classification head to use')
        parser.add_argument('--n-best-size',
                            default=5,
                            help='n best size for predictions')
        parser.add_argument('--start-n-top',
                            default=5,
                            help='Beam size for span start')
        parser.add_argument('--end-n-top',
                            default=5,
                            help='Beam size for span end')
        # fmt: on
        
    def get_masked_data(self, data, mask):
        return data * mask+MIN_FLOAT * (1-mask)
    
    def tile(self, data, size):
        for dim in range(-1, -1*len(size)-1, -1):
            multiple_num = size[dim]
            ori_data = data
            for _ in range(multiple_num-1):
                data = torch.cat([data, ori_data], dim=dim)
        return data

    def forward(self, model, sample, reduce=True): ####fairseq_task.py 430줄
        """Compute ranking loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        def compute_loss(label, predict, predict_mask, label_smoothing=0.0):
            masked_predict = self.get_masked_data(predict, predict_mask)
            masked_predict = predict #[b,l]
            
            if label_smoothing > 1e-10:
                onehot_label = F.one_hot(label, masked_predict.size(-1))
                onehot_label = (onehot_label * (1-label_smoothing) +
                                label_smoothing / masked_predict.size(-1).FloatTensor()) * predict_mask
                
                log_likelihood = F.log_softmax(masked_predict, dim=-1)
                loss = - (onehot_label*log_likelihood).sum(-1)
            else:
                CEL = torch.nn.CrossEntropyLoss()
                loss = CEL(masked_predict, label)
                
            return loss
        
        assert (
            hasattr(model, "classification_heads")
            and self.ranking_head_name in model.classification_heads
        ), "model must provide sentence ranking head for --criterion=coqa"

        logits, _ = model(
            sample["net_input"],
            classification_head_name=self.ranking_head_name,
        )
        
        p_mask = sample["net_input"]["p_mask"]
        preds = {}
        target_exist = sample["start_position"]!=None
        
        ##start
        start_result = logits["start_result"]
        sample_size = start_result.size()[0]
        start_result_mask = 1-p_mask
        
        start_result = torch.squeeze(start_result, dim=-1)
        start_result = self.get_masked_data(start_result, start_result_mask)
        start_prob = F.softmax(start_result, dim=-1)
        
        if not self.training:
            start_top_prob, start_top_index = torch.topk(start_prob, k=self.start_n_top)
            preds["start_prob"] = start_top_prob
            preds["start_index"] = start_top_index
            
        ##end
        end_result = logits["end_result"]
        if self.training:
            end_result_mask = 1-p_mask
            
            end_result = torch.squeeze(end_result, dim=-1)
            end_result = self.get_masked_data(end_result, end_result_mask)
            end_prob = F.softmax(end_result, dim=-1)
        else:
            end_result_mask = torch.unsqueeze(1-p_mask, dim=1)
            end_result_mask = self.tile(end_result_mask, (1, self.start_n_top, 1))
            
            end_result = torch.transpose(torch.squeeze(end_result, dim=-1), 1, 2)
            end_result = self.get_masked_data(end_result, end_result_mask)
            end_prob = F.softmax(end_result, dim=-1)
            
            end_top_prob, end_top_index = torch.topk(end_prob, k=self.start_n_top)
            preds["end_prob"] = end_top_prob
            preds["end_index"] = end_top_index
        
        ##unk
        unk_result = logits["unk_result"]
        unk_result_mask = torch.max(1-p_mask, dim=-1).values
        
        unk_result = torch.squeeze(unk_result, dim=-1)
        unk_result = self.get_masked_data(unk_result, unk_result_mask)
        unk_prob = F.sigmoid(unk_result)
        preds["unk_prob"] = unk_prob
        
        ##yes
        yes_result = logits["yes_result"]
        yes_result_mask = torch.max(1-p_mask, dim=-1).values
        
        yes_result = torch.squeeze(yes_result, dim=-1)
        yes_result = self.get_masked_data(yes_result, yes_result_mask)
        yes_prob = F.sigmoid(yes_result)
        preds["yes_prob"] = yes_prob
        
        ##no
        no_result = logits["no_result"]
        no_result_mask = torch.max(1-p_mask, dim=-1).values
        
        no_result = torch.squeeze(no_result, dim=-1)
        no_result = self.get_masked_data(no_result, no_result_mask)
        no_prob = F.sigmoid(no_result)
        preds["no_prob"] = no_prob
            
        ##num
        num_result = logits["num_result"]
        num_result_mask = torch.max(1-p_mask, dim=-1, keepdim=True).values
        
        num_result = self.get_masked_data(num_result, num_result_mask)
        num_probs = F.softmax(num_result, dim=-1)
        preds["num_probs"] = num_probs
        
        ##opt
        opt_result = logits["opt_result"]
        opt_result_mask = torch.max(1-p_mask, dim=-1, keepdim=True).values
        
        opt_result = self.get_masked_data(opt_result, opt_result_mask)
        opt_probs = F.softmax(opt_result, dim=-1)
        preds["opt_probs"] = opt_probs
        
        
        if target_exist and self.training:
            start_label = sample["start_position"]
            start_loss = compute_loss(start_label, start_result, 1-p_mask) # [b],[b,l],[b,l]
            end_label = sample["end_position"]
            end_loss = compute_loss(end_label, end_result, 1-p_mask) # [b], [b,l], [b,l]
            loss = torch.mean(start_loss + end_loss)
            
            unk_label = sample["is_unk"]
            unk_loss = F.binary_cross_entropy_with_logits(unk_result, unk_label.half())
            loss += torch.mean(unk_loss)
            
            yes_label = sample["is_yes"]
            yes_loss = F.binary_cross_entropy_with_logits(yes_result, yes_label.half())
            loss += torch.mean(yes_loss)
            
            no_label = sample["is_no"]
            no_loss = F.binary_cross_entropy_with_logits(no_result, no_label.half())
            loss += torch.mean(no_loss)
            
            num_label = sample["number"]
            num_result_mask = torch.max(1-p_mask, dim=-1, keepdim=True).values
            num_loss = compute_loss(num_label, num_result, num_result_mask)
            loss += torch.mean(num_loss)
            
            opt_label = sample["option"]
            opt_result_mask = torch.max(1-p_mask, dim=-1, keepdim=True).values
            opt_loss = compute_loss(opt_label, opt_result, opt_result_mask)
            loss += torch.mean(opt_loss)
            targets = sample
            
        elif target_exist:
            start_label = sample["start_position"]
            start_loss = compute_loss(start_label, start_result, 1-p_mask) # [b],[b,l],[b,l]
            end_label = sample["end_position"]
            end_result = end_result[:,0,:]
            end_loss = compute_loss(end_label, end_result, 1-p_mask) # [b],[b,k,l],[b,l]
            loss = torch.mean(start_loss + end_loss)
            unk_label = sample["is_unk"]
            unk_loss = F.binary_cross_entropy_with_logits(unk_result, unk_label.half())
            loss += torch.mean(unk_loss)
            
            yes_label = sample["is_yes"]
            yes_loss = F.binary_cross_entropy_with_logits(yes_result, yes_label.half())
            loss += torch.mean(yes_loss)
            
            no_label = sample["is_no"]
            no_loss = F.binary_cross_entropy_with_logits(no_result, no_label.half())
            loss += torch.mean(no_loss)
            
            num_label = sample["number"]
            num_result_mask = torch.max(1-p_mask, dim=-1, keepdim=True).values
            num_loss = compute_loss(num_label, num_result, num_result_mask)
            loss += torch.mean(num_loss)
            
            opt_label = sample["option"]
            opt_result_mask = torch.max(1-p_mask, dim=-1, keepdim=True).values
            opt_loss = compute_loss(opt_label, opt_result, opt_result_mask)
            loss += torch.mean(opt_loss)
            targets = sample            
            
        else:
            loss = torch.tensor(0.0, requires_grad=True)
            targets = None
            
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        if self.prediction_h is not None and not self.training:
            predictions = []
            for i in range(sample["nsentences"]):
                pred = {}
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
                prediction = json.dumps(pred)
                predictions.append(prediction)
                #self.prediction_h.write(prediction)
                #self.prediction_h.write("\n")
            return loss, predictions, sample_size, logging_output

        return loss, sample_size, logging_output

    ###한번의 batch마다 불러짐
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
