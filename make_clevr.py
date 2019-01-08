import os
from collections import defaultdict
import numpy as np
import json
import re
import pickle
import torch
from pathlib import Path
home = str(Path.home())

question_type_dict = {'exist': 10,
                    'count': 20,
                    'equal_integer': 30,
                    'less_than': 31,
                    'greater_than': 32,
                    'query_size': 40,
                    'query_color': 41,
                    'query_material': 42,
                    'query_shape': 43,
                    'equal_size': 50,
                    'equal_color': 51,
                    'equal_material': 52,
                    'equal_shape': 53
                    }


def make_data(data_dir):
    print("Start making data pickle")
    q_corpus = set()
    a_corpus = set()
    modes = ['train', 'val']
    q_list = dict()
    qa_list = defaultdict(list)
    for mode in modes:
        qf_dir = os.path.join(data_dir, 'questions', 'CLEVR_{}_questions.json'.format(mode))
        with open(qf_dir) as f:
            q_list[mode] = json.load(f)['questions']
        for q_obj in q_list[mode]:
            img_f = q_obj['image_filename']
            q_text = q_obj['question'].lower()
            q_words = re.sub('[^A-Za-z ]+', "", q_text).split(' ')
            q_corpus.update(q_words)
            a_text = str(q_obj['answer']).lower().strip()
            a_corpus.add(a_text)
            q_type = question_type_dict[q_obj['program'][-1]['type']]
            qa_list[mode].append((img_f, q_words, a_text, q_type))
    word_to_idx = {"<pad>": 0, "<eos>": 1}
    idx_to_word = {0: "<pad>", 1: "<eos>"}
    answer_word_to_idx = dict()
    answer_idx_to_word = dict()
    for word in q_corpus:
        word_to_idx[word] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = word
    for word in a_corpus:
        answer_word_to_idx[word] = len(answer_word_to_idx)
        answer_idx_to_word[len(answer_idx_to_word)] = word
    data_dict = {'question': {'word_to_idx': word_to_idx,
                              'idx_to_word': idx_to_word},
                 'answer': {'word_to_idx': answer_word_to_idx,
                            'idx_to_word': answer_idx_to_word},
                 'question_type': question_type_dict}
    with open(os.path.join(data_dir, 'data_dict.pkl'), 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('data_dict.pkl saved')
    qa_idx_data = defaultdict(list)
    for mode in modes:
        for img_file, q_word_list, answer_word, q_type in qa_list[mode]:
            q = [word_to_idx[word] for word in q_word_list]
            q.append(1)
            q = torch.from_numpy(np.array(q))
            a = answer_word_to_idx[answer_word]
            a = torch.from_numpy(np.array(a)).view(1)
            q_t = torch.from_numpy(np.array(q_type)).view(1)
            qa_idx_data[mode].append((img_file, q, a, q_t))
        with open(os.path.join(data_dir, 'data_{}.pkl'.format(mode)), 'wb') as file:
            pickle.dump(qa_idx_data[mode], file, protocol=pickle.HIGHEST_PROTOCOL)
        print('data_{}.pkl saved'.format(mode))


if __name__ =='__main__':
    data_directory = os.path.join(home, 'data/sample')
    make_data(data_directory)
