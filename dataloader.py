from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import pickle
import os
import pandas as pd
import numpy as np
from PIL import Image
from imageio import imread
import json
from time import time
import re
from collections import defaultdict

def collate_text(list_inputs):
    list_inputs.sort(key=lambda x:len(x[1]), reverse = True)
    images = torch.Tensor()
    questions = []
    answers = torch.Tensor().to(torch.long)
    for i, q, a in list_inputs:
        images = torch.cat([images, i.unsqueeze(0)], 0)
        questions.append(q)
        answers = torch.cat([answers, a], 0)
    questions_packed = pack_sequence(questions)
    return images, questions_packed, answers

def train_loader(data, data_directory = '/home/sungwonlyu/data/', batch_size = 128, input_h = 128, input_w = 128):
    if data == 'clevr':
        train_dataloader = DataLoader(
            Clevr(data_directory + data + '/', train=True, 
            transform = transforms.Compose([transforms.Resize((input_h, input_w)),
                                                transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True,
            collate_fn = collate_text)
    return train_dataloader

def test_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128, input_h = 128, input_w = 128):
    if data == 'clevr':
        test_dataloader = DataLoader(
            Clevr(data_directory + data + '/', train=False, 
            transform = transforms.Compose([transforms.Resize((input_h, input_w)),
                                                transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True,
            collate_fn = collate_text)
    return test_dataloader


class Clevr(Dataset):
    """Clevr dataset."""
    def __init__(self, root_dir, train = True, transform = None):
        self.root_dir = root_dir
        # self.mode = 'sample'
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.q_dir = self.root_dir + 'questions/'+ 'CLEVR_{}_questions.json'.format(self.mode)
        self.img_dir = self.root_dir + 'images/'+ '{}/'.format(self.mode)
        if self.mode == 'sample':
            self.img_dir = self.root_dir + 'images/train/'
        self.sample = False
        self.load_data()

    def make_data(self):
        q_corpus = set()
        a_corpus = set()
        # if self.sample:
        #     modes = ['sample']
        # else:
        modes = ['train', 'val', 'sample']
        q_list = dict()
        qa_list = defaultdict(list)
        for mode in modes:
            # if self.sample:
            #     img_dir = self.root_dir + 'images/train/'
            #     ann_dir = self.root_dir + 'questions/CLEVR_{}_questions.json'.format(mode)
            # else:
            img_dir = self.root_dir + 'images/{}/'.format(mode)
            if mode == 'sample':
                img_dir = self.root_dir + 'images/train/'
            ann_dir = self.root_dir + 'questions/CLEVR_{}_questions.json'.format(mode)
            with open(ann_dir) as f:
                q_list[mode] = json.load(f)['questions']
            for q_obj in q_list[mode]:
                img_dir = q_obj['image_filename']
                q_text = q_obj['question'].lower()
                q_text = re.sub('\s+', ' ', q_text)
                q_text_without_question_mark = q_text[:-1]
                q_words = q_text_without_question_mark.split(' ')
                q_words.insert(0, 1)
                q_words.append(2)
                q_corpus.update(q_words)
                a_text = q_obj['answer'].lower()
                a_text = re.sub('\s+', ' ', a_text)
                a_corpus.add(a_text)
                qa_list[mode].append((img_dir, q_words, a_text))
                # q_corpus.update(a_corpus)

        word_to_idx = {"PAD":0, "SOS": 1, "EOS": 2}
        idx_to_word = {0: "PAD", 1: "SOS", 2: "EOS"}
        answer_word_to_idx = dict()
        answer_idx_to_word = dict()
        for idx, word in enumerate(q_corpus, start=3):
            # index starts with 1 because 0 is used as the padded value when batches are
            #  created
            word_to_idx[word] = idx
            idx_to_word[idx] = word

        for idx, word in enumerate(a_corpus):
            answer_word_to_idx[word] = idx
            answer_idx_to_word[idx] = word
        #     # single answer, so no padded values of 0 are created. thus index starts with 0
        data_dict = {'question': {'word_to_idx' : word_to_idx,
                                    'idx_to_word' : idx_to_word},
                        'answer': {'word_to_idx' : answer_word_to_idx,
                                    'idx_to_word' : answer_idx_to_word}}
        with open('data_dict.pkl', 'wb') as file:
            pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('data_dict.pkl saved')

        qa_idx_data = defaultdict(list)
        for mode in modes:
            for img_dir, q_word_list, answer_word in qa_list[mode]:                  
                q = [word_to_idx[word] for word in q_word_list]
                q = torch.from_numpy(np.array(q))
                a = answer_word_to_idx[answer_word]
                a = torch.from_numpy(np.array(a)).view(1)
                qa_idx_data[mode].append((img_dir, q, a))
            with open('qa_idx_data_{}.pkl'.format(mode), 'wb') as file:
                pickle.dump(qa_idx_data[mode], file, protocol=pickle.HIGHEST_PROTOCOL)
            print('qa_idx_data_{}.pkl saved'.format(mode))

    def load_data(self):
        with open('qa_idx_data_{}.pkl'.format(self.mode), 'rb') as file:
            self.qa_idx_data = pickle.load(file)
        with open('data_dict.pkl', 'rb') as file:
            self.data_dict = pickle.load(file)
        self.word_to_idx = self.data_dict['question']['word_to_idx']
        self.idx_to_word = self.data_dict['question']['idx_to_word']
        self.answer_word_to_idx = self.data_dict['answer']['word_to_idx']
        self.answer_idx_to_word = self.data_dict['answer']['idx_to_word']
        self.q_size = len(self.word_to_idx)
        self.a_size = len(self.answer_word_to_idx)

    def __len__(self):
        return len(self.qa_idx_data)

    def __getitem__(self, idx):
        img_dir, q, a = self.qa_idx_data[idx]
        image = Image.open(self.img_dir + img_dir).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, q, a

def debug():
    start = time()
    train_dataloader = train_loader('clevr', batch_size = 4)
    b = time() - start
    print(b)
    start = time()
    sum_ = 0
    n = 0
    for i, q, a in train_dataloader:
        n += 1
        # print(i)
        # print(q) 
        # print(a)
        b = time() - start
        print(b)
        sum_ += b   
        start = time()
    print(sum_)
def make_data():
    train_dataloader = train_loader('clevr', batch_size = 4)
    train_dataloader.dataset.make_data()
# make_data()
# debug()
