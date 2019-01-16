import os
from collections import defaultdict
import numpy as np
import json
import re
import pickle
import torch
from pathlib import Path
home = str(Path.home())


def make_data(data_dir):
    print("Start making VQA 2.0 data pickle")
    q_corpus = set()
    a_corpus = set()
    question_types = set()
    # answer_types = set()
    modes = ['train', 'val']
    qa_list = defaultdict(list)
    question_list = {}
    for mode in modes:
        question_file = os.path.join(data_dir, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(mode))
        with open(question_file) as f:
            questions = json.load(f)["questions"]
        for question in questions:
            question_list[question['question_id']] = question['question']
        annotation_file = os.path.join(data_dir, 'v2_mscoco_{}2014_annotations.json'.format(mode))
        with open(annotation_file) as f:
            annotations = json.load(f)["annotations"]
        for q_obj in annotations:
            image_id = q_obj['image_id']
            question_text = question_list[q_obj['question_id']]
            question_words = re.sub('[^0-9A-Za-z ]+', "", question_text).lower().split(' ')
            question_type = q_obj['question_type']
            answer_word = q_obj["multiple_choice_answer"]
            answer_type = q_obj["answer_type"]
            # question_type_idx = question_type_dict[question_type]
            # answer_type_idx = answer_type_dict[answer_type]
            q_corpus.update(question_words)
            a_corpus.add(answer_word)
            question_types.add(question_type)
            question_types.add(answer_type)
            qa_list[mode].append((image_id, question_words, answer_word, question_type, answer_type))
    word_to_idx = {"<pad>": 0, "<eos>": 1}
    idx_to_word = {0: "<pad>", 1: "<eos>"}
    answer_word_to_idx = dict()
    answer_idx_to_word = dict()
    question_type_to_idx = dict()
    idx_to_question_type = dict()
    # answer_type_to_idx = dict()
    # idx_to_answer_type = dict()
    for word in sorted(list(q_corpus)):
        word_to_idx[word] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = word
    for word in sorted(list(a_corpus)):
        answer_word_to_idx[word] = len(answer_word_to_idx)
        answer_idx_to_word[len(answer_idx_to_word)] = word
    for question_type in sorted(list(question_types)):
        question_type_to_idx[question_type] = len(question_type_to_idx)
        idx_to_question_type[len(idx_to_question_type)] = question_type
    print(len(q_corpus), len(a_corpus), len(question_types))
    # for answer_type in sorted(list(answer_types)):
    #     answer_type_to_idx[answer_type] = len(answer_type_to_idx)
        # idx_to_answer_type[len(idx_to_answer_type)] = answer_type
    data_dict = {'word_to_idx': word_to_idx,
                 'idx_to_word': idx_to_word,
                 'answer_word_to_idx': answer_word_to_idx,
                 'answer_idx_to_word': answer_idx_to_word,
                 'question_type_to_idx': question_type_to_idx,
                 'idx_to_question_type': idx_to_question_type
                 # 'answer_type_to_idx': answer_type_to_idx,
                 # 'idx_to_answer_type': idx_to_answer_type
                 }
    with open(os.path.join(data_dir, 'data_dict.pkl'), 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('data_dict.pkl saved')
    qa_idx_data = defaultdict(list)
    for mode in modes:
        for image_id, question_words, answer_word, question_type, answer_type in qa_list[mode]:
            q = [word_to_idx[word] for word in question_words]
            q.append(1)
            q = torch.from_numpy(np.array(q))
            a = answer_word_to_idx[answer_word]
            a = torch.from_numpy(np.array(a)).view(1)
            q_t = torch.from_numpy(np.array(question_type_to_idx[question_type])).view(1)
            a_t = torch.from_numpy(np.array(question_type_to_idx[answer_type])).view(1)
            qa_idx_data[mode].append((image_id, q, a, q_t, a_t))
        print(len(qa_idx_data[mode]))
        with open(os.path.join(data_dir, 'data_{}.pkl'.format(mode)), 'wb') as file:
            pickle.dump(qa_idx_data[mode], file, protocol=pickle.HIGHEST_PROTOCOL)
        print('data_{}.pkl saved'.format(mode))


if __name__ =='__main__':
    data_directory = os.path.join(home, 'data/vqa2')
    make_data(data_directory)
