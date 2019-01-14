import os
from collections import defaultdict
import numpy as np
import json
import re
import pickle
import torch
from pathlib import Path
home = str(Path.home())

# question_type_dict = {'what is': 10,
#                     'what color': 11,
#                     'what kind': 12,
#                     'what are': 13,
#                     'what does': 14,
#                     'what time': 15,
#                     'what sport': 16,
#                     'what animal': 17,
#                     'what brand': 18,
#                     'is the': 20,
#                     'is this': 21,
#                     'is there': 22,
#                     'how many': 30,
#                     'are': 40,
#                     'does': 50,
#                     'where': 60,
#                     'why': 70,
#                     'which': 80,
#                     'do': 90,
#                     'who': 100
#                     }
#
# answer_type_dict = {'yes/no': 0,
#                     'number': 1,
#                     'other': 2
#                     }

def make_data(data_dir):
    print("Start making VQA 2.0 data pickle")
    corpus = set()
    question_types = set()
    answer_types = set()
    modes = ['train', 'val']
    qa_list = defaultdict(list)
    question_list = {}
    for mode in modes:
        question_file = os.path.join(data_dir, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(mode))
        with open(question_file) as f:
            questions = json.load(f)["questions"]
        print(len(questions))
        for question in questions:
            question_list[question['question_id']] = question['question']
        annotation_file = os.path.join(data_dir, 'v2_mscoco_{}2014_annotations.json'.format(mode))
        with open(annotation_file) as f:
            annotations = json.load(f)["annotations"]
        print(len(annotations))
        for q_obj in annotations:
            image_id = q_obj['image_id']
            question_text = question_list[q_obj['question_id']]
            question_words = re.sub('[^0-9A-Za-z ]+', "", question_text).lower().split(' ')
            question_type = q_obj['question_type']
            answer_text = q_obj["multiple_choice_answer"]
            answer_words = re.sub('[^0-9A-Za-z ]+', "", str(answer_text)).lower().split(' ')
            answer_type = q_obj["answer_type"]
            # question_type_idx = question_type_dict[question_type]
            # answer_type_idx = answer_type_dict[answer_type]
            corpus.update(question_words)
            corpus.update(answer_words)
            question_types.add(question_type)
            answer_types.add(answer_type)
            qa_list[mode].append((image_id, question_words, answer_words, question_type, answer_type))
    word_to_idx = {"<pad>": 0, "<eos>": 1}
    idx_to_word = {0: "<pad>", 1: "<eos>"}
    question_type_to_idx = dict()
    idx_to_question_type = dict()
    answer_type_to_idx = dict()
    idx_to_answer_type = dict()
    for word in corpus:
        word_to_idx[word] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = word
    for question_type in question_types:
        question_type_to_idx[question_type] = len(question_type_to_idx)
        idx_to_question_type[len(idx_to_question_type)] = question_type
    for answer_type in answer_types:
        answer_type_to_idx[answer_type] = len(answer_type_to_idx)
        idx_to_answer_type[len(idx_to_answer_type)] = answer_type
    data_dict = {'word_to_idx': word_to_idx,
                 'idx_to_word': idx_to_word,
                 'question_type_to_idx': question_type_to_idx,
                 'idx_to_question_type': idx_to_question_type,
                 'answer_type_to_idx': answer_type_to_idx,
                 'idx_to_answer_type': idx_to_answer_type}
    with open(os.path.join(data_dir, 'data_dict.pkl'), 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('data_dict.pkl saved')
    qa_idx_data = defaultdict(list)
    for mode in modes:
        for image_id, question_words, answer_words, question_type, answer_type in qa_list[mode]:
            q = [word_to_idx[word] for word in question_words]
            q.append(1)
            q = torch.from_numpy(np.array(q))
            a = [word_to_idx[word] for word in answer_words]
            a.append(1)
            a = torch.from_numpy(np.array(a))
            q_t = torch.from_numpy(np.array(question_type_to_idx[question_type])).view(1)
            a_t = torch.from_numpy(np.array(answer_type_to_idx[answer_type])).view(1)
            qa_idx_data[mode].append((image_id, q, a, q_t, a_t))
        with open(os.path.join(data_dir, 'data_{}.pkl'.format(mode)), 'wb') as file:
            pickle.dump(qa_idx_data[mode], file, protocol=pickle.HIGHEST_PROTOCOL)
        print('data_{}.pkl saved'.format(mode))


if __name__ =='__main__':
    data_directory = os.path.join(home, 'data/vqa2')
    make_data(data_directory)
