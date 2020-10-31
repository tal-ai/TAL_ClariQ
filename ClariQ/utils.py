import os
import random
import pickle
import pandas as pd
import numpy as np
# import math, copy, time
# import heapq, random, copy
import os
import torch

def concat_input(request, context,max_word=81):
    one_text = '<sos> ' + request.lower()
    if len(context) == 0:
        pass
    else:
        for one_c in context:
            one_text += ' <sep> ' + one_c['question'].lower() + ' <sep> ' + one_c['answer'].lower()
    splited_one_text = one_text.split(' ')
    if len(splited_one_text) > max_word:
        splited_one_text = splited_one_text[len(splited_one_text)-max_word:]
        return ' '.join(splited_one_text)
    else:
        return one_text


class Constants:
    def __init__(self):
        self.BOS_WORD = '<sos>'
        self.EOS_WORD = '<eos>'
        self.PAD_WORD = '<blank>'
        self.UNK_WORD = '<unk>'
        self.SEP_WORD = '<sep>'
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.SEP = 4
Constants = Constants()

def convert_insts_to_num(dia_insts, word2idx):
    res =[]
    for word_insts in dia_insts:
        this_res = []
        for w in word_insts:
            this_res.append(word2idx.get(w,Constants.UNK))
        res.append(this_res)
    return res

def convert_inst_to_num(one_sent, word2idx):
    one_sent = one_sent.split(' ')
    return [word2idx.get(w, Constants.UNK) for w in one_sent]

def process_data(request, question, contexts, vocab):
    input_left = concat_input(request, contexts).replace('.','').replace('?','')
    input_right = question.lower()
    left_id = convert_inst_to_num(input_left, vocab)
    right_id = convert_inst_to_num(input_right, vocab)
    return left_id, right_id

def query_one_batch_data(query, que_bank, vocab):
    query = query.replace('.','').replace('?','')
    query_list = query.split(' ')
    input_left = [query_list] * len(que_bank)
    input_right_ids = que_bank
    input_left_ids = convert_insts_to_num(input_left, vocab)
#     input_right_ids = convert_inst_to_num(input_right, vocab)
    return input_left_ids, input_right_ids

def pad_sequences(input_insts, maxlen, value):
    new_insts = []
    for inst in input_insts:
        if len(inst) <= maxlen:
            inst += [value] * (maxlen-len(inst))
        else:
            inst = inst[:maxlen]
        new_insts.append(inst)
    return new_insts

def collate_clair_batch(left, right):
    q_len_left = [len(x) for x in left]
    q_len_right = [len(y) for y in right]
    
    max_len_left = max(q_len_left)
    max_len_right = max(q_len_right)
    
    input_ids_left = pad_sequences(left, maxlen=max_len_left, value=0)
    input_ids_right = pad_sequences(right, maxlen=max_len_right, value=0)
    return torch.LongTensor(input_ids_left), torch.LongTensor(input_ids_right)

def change_id_to_string(qid):
    str_qid = str(qid)
    while len(str_qid)<4:
        str_qid = '0' + str_qid
    return 'Q0' + str_qid

def select_no_duplicate_questions(q_list, conv_context, id_to_text):
    qtext_list = qid_to_question(q_list, id_to_text)
#     print(qtext_list)
    prev_questions = [x['question'] for x in conv_context]
#     print(bm25_preds)
    pred_list = []
    for q in qtext_list:
        if q not in prev_questions:
            pred_list.append(q)
    return pred_list

def qid_to_question(qid_list, dict_qid):
    question_list = []
    for qid in qid_list:
        question_list.append(dict_qid[qid])
    return question_list



