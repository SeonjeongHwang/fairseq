import sys
import collections
import os
import os.path
import string

def get_QuAC_examples(data_dir, Train=True, num_turns=2):
    
    data_pipeline = QuacPipeline(data_dir=data_dir, num_turn=num_turn)
    
    if Train:
        train_examples = data_pipeline.get_train_examples()
        return train_examples
    else:
        predict_examples = data_pipeline.get_dev_examples()
        return predict_examples

class QuacPipeline(object):
    """Pipeline for QuAC dataset."""
    def __init__(self,
                 data_dir,
                 task_name,
                 num_turn):
        self.data_dir = data_dir
        self.num_turn = num_turn
    
    #2
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-quac.json")
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    #2
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-quac.json")
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    #3
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)["data"]
                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(data_path))
    
    def _get_question_text(self,
                           history,
                           qas):
        question_tokens = ['<s>'] + qas["question"].split(' ')
        return " ".join(history + [" ".join(question_tokens)])
    
    def _get_question_history(self,
                              history,
                              qas,
                              num_turn):
        question_tokens = []
        question_tokens.extend(['<s>'] + qas["question"].split(' '))
        question_tokens.extend(['</s>'] + qas["orig_answer"]["text"].split(' '))
        
        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)
        
        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]
        
        return history
    
    def _get_answer_span(self,
                         context,
                         qas,
                         no_answer):
        orig_text = qas["orig_answer"]["text"].lower()
        answer_start = qas["orig_answer"]["answer_start"]
        
        if no_answer or not orig_text or answer_start < 0:
            return "", -1, -1
        
        answer_end = answer_start + len(orig_text) - 1
        answer_text = context[answer_start:answer_end + 1].lower()
        
        assert orig_text == answer_text
        answer_text = context[answer_start:answer_end + 1]
        
        return answer_text, answer_start, answer_end
    
    #4
    def _get_example(self,
                     data_list):
        examples = []
        for data in data_list:
            for paragraph in data["paragraphs"]:
                data_id = paragraph["id"]
                paragraph_text = paragraph["context"]
                if paragraph_text.endswith("CANNOTANSWER"):
                    paragraph_text = paragraph_text[:-len("CANNOTANSWER")].rstrip()
                
                question_history = []
                for qas in paragraph["qas"]:
                    qas_id = qas["id"]
                    
                    question_text = self._get_question_text(question_history, qas)
                    question_history = self._get_question_history(question_history, qas, self.num_turn)
                    
                    no_answer = (qas["orig_answer"]["text"] == "CANNOTANSWER")
                    orig_answer_text, start_position, _ = self._get_answer_span(paragraph_text, qas, no_answer)
                    
                    yes_no = qas["yesno"]
                    follow_up = qas["followup"]
                    
                    example = {
                        'qas_id':qas_id,
                        'question_text':question_text,
                        'paragraph_text':paragraph_text,
                        'orig_answer_text':orig_answer_text,
                        'start_position':start_position,
                        'no_answer':no_answer,
                        'yes_no':yes_no,
                        'follow_up':follow_up}
                    examples.append(example)
        return examples
