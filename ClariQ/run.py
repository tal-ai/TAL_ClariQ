# -*- coding: utf-8 -*-
import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
import random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from model.CMAN import MwAN_trans
# import math, copy, time
# import heapq, copy
# import logging

# device is cpu if no gpu available
device = torch.device("cuda")
# load vocabulary(a binary file)
vocabulary = torch.load(os.path.join(base_dir, "processed_data", "vocab.pt"))
# load question bank
question_bank_path = os.path.join(os.path.dirname(__file__), "processed_data","question_bank.tsv")
question_bank = pd.read_csv(question_bank_path, sep='\t').fillna('')
question_bank['tokenized_question'] = question_bank['question'].map(lambda x: x.lower().split(' '))
qid_to_text = {}
for idx, row in question_bank.iterrows():
    qid_to_text[row['question_id']] = row['question']
lower_questions = []
for one_que in question_bank.question.tolist():
    lower_questions.append(one_que.lower().split(' '))
converted_lower_questions = convert_insts_to_num(lower_questions, vocabulary)

# load model
model_path = os.path.join(os.path.dirname(__file__), "saved_model","CMAN3.chkpt")
emb_path = os.path.join(os.path.dirname(__file__), "processed_data","pretrained_emb.pt")
pretrained_emb = torch.load(emb_path)
model = MwAN_trans(
    d_model=100, 
    number_block=2, 
    head_number=4, 
    d_ff=400, 
    seq_len=192, 
    vocab_size=len(vocabulary),
    pretrained=pretrained_emb,
    drop_out=0.1)
checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
print('[Info] Trained model state loaded.')
model = model.to(device)
model = model.eval()

class ClariQ:
    def __init__(self):
        self.model = model
        # self.model.eval()
    
    def get_query_result(self, model, left_tensor, right_tensor):
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        pred_res,_,_ = model(left_tensor, right_tensor)
        pred_res = torch.softmax(pred_res, dim=-1)[:,1]
        pred_prob = pred_res.cpu().numpy()
        return pred_prob

    def forward(self, input_dict, batch_size):
        request = input_dict['initial_request']
        context = input_dict['conversation_context']
        input_left = concat_input(request, context)
        batch_left, batch_right = query_one_batch_data(input_left, converted_lower_questions, vocabulary)
        batch_left, batch_right = collate_clair_batch(batch_left, batch_right)
        whole_pred_prob = []
        with torch.no_grad():
            for split in range(0, len(batch_left), batch_size):
                left_2k, right_2k = batch_left[split:split+batch_size], batch_right[split:split+batch_size]
                query_pred_tmp = self.get_query_result(self.model, left_2k, right_2k)
                whole_pred_prob.append(query_pred_tmp)
        query_pred_prob = np.concatenate(whole_pred_prob)
        query_qid = np.argsort(query_pred_prob)[::-1] + 1 # since index start from 0
        str_qid = list(map(change_id_to_string, query_qid))
        return str_qid

def example():
    import time
    t0 = time.time()
    cq_query = ClariQ()
    one_input_case = {'topic_id': 8,
        'facet_id': 'F0968',
        'initial_request': 'I want to know about appraisals.',
        'question': 'are you looking for a type of appraiser',
        'answer': 'im looking for nearby companies that do home appraisals',
        'conversation_context': [],
        'context_id': 968}
    # pred_qid = cq_query.forward(one_input_case)
    qid_list = query_one_dict(one_input_case)
    t1 = time.time()
    # print("Query ID: ",pred_qid[0])
    print("Query result: ", qid_list[0])
    print("time: %0.4f ms" % ((t1 - t0) * 1000))


def query_one_dict(input_dict, batch_size=1000):
    """
    input:
        type: dict
        case: {'topic_id': 8,
        'facet_id': 'F0968',
        'initial_request': 'I want to know about appraisals.',
        'question': 'are you looking for a type of appraiser',
        'answer': 'im looking for nearby companies that do home appraisals',
        'conversation_context': [],
        'context_id': 968}
    return: 
        type: string
        case: would you like to see a price range for appraisals
        note: output one question from the question bank 

    This is a function for query question from one input dict
    """
    cq_query = ClariQ()
    # if dialogue round reach to 5, automatic return Q00001
    dialogue_ctx = input_dict['conversation_context']
    response = ''
    if len(dialogue_ctx) == 5:
        q_res_id = 'Q00001'
        response = qid_to_text['Q00001']
    else:
        pred_qid = cq_query.forward(input_dict, batch_size)
        real_preds = select_no_duplicate_questions(pred_qid[:10], input_dict['conversation_context'], qid_to_text)
        # q_list = qid_to_question(real_preds, qid_to_text) # return top one from the pred
        response = real_preds[0]
    return response


def write_test_file(multi_turn_request_file_path, output_run_file, topk=100, batch_size=1000):
    """
    input:
        multi_turn_request_file_path
            type: string
            explanation: absolute path for input test file
        output_run_file
            type: string
            explanation: output file for input test cases, this function will 
                        automatically write all the query result in the output
                        file
        topk
            type: int
            explanation: we write topk number of questions for one query context id

    This is a function to write test file which is used to calculated NDCG, Precision and MRR score.
    We write 
    """
    with open(multi_turn_request_file_path, 'rb') as fi:
        test = pickle.load(fi)
    # Reads the dev file and create the context_dict to make predictions
    cq_query = ClariQ()

    context_dict = dict()
    for rec_id in test:
        ctx_id = test[rec_id]['context_id']
        if ctx_id not in context_dict:
            context_dict[ctx_id] = {'initial_request': test[rec_id]['initial_request'],
                                'conversation_context': test[rec_id]['conversation_context'],
                               }
    
    test_query_res = {}
    for rec_id in test:
        input_dict = test[rec_id]
        ctx_id = input_dict['context_id']
        if ctx_id not in test_query_res:
            pred_qid = cq_query.forward(input_dict, batch_size)
            # q_list = qid_to_question(pred_qid, qid_to_text)
            # test_query_res[ctx_id] = q_list
            # real_preds = select_no_duplicate_questions(pred_qid[:topk], input_dict['conversation_context'], qid_to_text)
            # q_list = qid_to_question(real_preds, qid_to_text) # return topk-unknow_number question, -unknown_number since we do not know how many questions have queried in context history
            test_query_res[ctx_id] = pred_qid
    
    # finish query, write file
    with open(output_run_file,'w') as f:
        for ctx_id in context_dict:
            _preds = select_no_duplicate_questions(test_query_res[ctx_id][:topk], context_dict[ctx_id]['conversation_context'], qid_to_text)
            for _i, _qid in enumerate(_preds):
                f.write('{} 0 "{}" {} {} CMAN_multi_turn\n'.format(ctx_id, _qid, _i, len(_preds)-_i))


if __name__ == "__main__":
    # example()
    case_0 = {'topic_id': 293,
        'facet_id': 'F0729',
        'initial_request': 'Tell me about the educational advantages of social networking sites.',
        'question': 'which social networking sites would you like information on',
        'answer': 'i don have a specific one in mind just overall educational benefits to social media sites',
        'conversation_context': [
            {'question': 'what level of schooling are you interested in gaining the advantages to social networking sites',
            'answer': 'all levels'},
            {'question': 'what type of educational advantages are you seeking from social networking',
            'answer': 'i just want to know if there are any'}
        ]
    }
    print(query_one_dict(case_0, 100))

    # multi_turn_request_file_path = "/share/Tianqiao_Liu/clairQ/processed_data/little_dev.pkl"
    # output_run_file = "/share/Tianqiao_Liu/clairQ/processed_data/run_file_dev"
    # write_test_file(multi_turn_request_file_path, output_run_file, 10)




