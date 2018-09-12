from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
from cv2 import imread
from collections import defaultdict
import json
from time import time
import re

def train_loader(data, data_directory = '/home/sungwonlyu/data/', batch_size = 128):
    if data == 'mnist':
        train_dataloader = DataLoader(
            datasets.MNIST(data_directory + data + '/', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'svhn':
        train_dataloader = DataLoader(
            datasets.SVHN(data_directory + data + '/', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'cifar10':
        train_dataloader = DataLoader(
            datasets.CIFAR10(data_directory + data + '/', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'alphachu':
        train_dataloader = DataLoader(
            AlphachuDataset(data_directory + data + '/', train=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'clevr':
        train_dataloader = DataLoader(
            Clevr(data_directory + data + '/', train=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    return train_dataloader

def test_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128):
    if data == 'mnist':
        test_dataloader = DataLoader(
            datasets.MNIST(data_directory + data + '/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'svhn':
        test_dataloader = DataLoader(
            datasets.SVHN(data_directory + data + '/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'cifar10':
        test_dataloader = DataLoader(
            datasets.CIFAR10(data_directory + data + '/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'alphachu':
        test_dataloader = DataLoader(
            AlphachuDataset(data_directory + data + '/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'clevr':
        test_dataloader = DataLoader(
            Clevr(data_directory + data + '/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    return test_dataloader


class Clevr(Dataset):
    """Clevr dataset."""
    def __init__(self, root_dir, train = True, transform = None):
        print(1)
        self.ckpt = time()
        self.root_dir = root_dir
        self.q_dir = self.root_dir + 'questions/'
        self.img_dir = self.root_dir + 'images/'
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.img_dir = self.img_dir + '{}/'.format(self.mode)
        self.ann_dir = self.q_dir + 'CLEVR_{}_questions.json'.format(self.mode)
        with open(self.ann_dir) as f:
            self.q_list = json.load(f)['questions']
        print(2, time() - self.ckpt)
        self.ckpt = time()
        self.make_corpus()

    def make_corpus(self):
        q_corpus = set()
        a_corpus = set()
        qa_data = []
        for q_obj in self.q_list:
            img_dir = q_obj['image_filename']
            q_text = q_obj['question'].lower()
            q_text = re.sub('\s+', ' ', q_text)
            q_text_without_question_mark = q_text[:-1]
            q_words = q_text_without_question_mark.split(' ')
            q_corpus.update(q_words)
            a_text = q_obj['answer'].lower()
            a_text = re.sub('\s+', ' ', a_text)
            a_corpus.add(a_text)
            qa_data.append((img_dir, q_words, a_text))
        self.q_size = len(q_corpus)
        self.a_size = len(a_corpus)
        print(3, time() - self.ckpt)
        self.ckpt = time()

        word_to_idx = {"SOS": 1, "EOS": 2}
        idx_to_word = {2: "SOS", 1: "EOS"}
        for idx, word in enumerate(q_corpus, start=3):
            # index starts with 1 because 0 is used as the padded value when batches are
            #  created
            word_to_idx[word] = idx
            idx_to_word[idx] = word
        print(4, time() - self.ckpt)
        self.ckpt = time()

        answer_word_to_idx = dict()
        answer_idx_to_word = dict()
        for idx, word in enumerate(a_corpus, start=0):
            # single answer, so no padded values of 0 are created. thus index starts with 0
            answer_word_to_idx[word] = idx
            answer_idx_to_word[idx] = word
        print(5, time() - self.ckpt)
        self.ckpt = time()

        self.qa_idx_data = []
        for img_dir, q_word_list, answer_word in qa_data:
            q = [word_to_idx[word] for word in q_word_list]
            q.insert(0, 1)
            q.append(2)
            a = answer_word_to_idx[answer_word]
            self.qa_idx_data.append((img_dir, q, a))
        print(6, time() - self.ckpt)
        self.ckpt = time()


    def __len__(self):
        return len(self.qa_idx_data)

    def __getitem__(self, idx):
        img_dir, q, a = self.qa_idx_data[idx]
        image = imread(self.img_dir + img_dir)
        image = image[:,:,::-1].transpose((2,0,1)).copy()
        image = torch.from_numpy(image).float().div(255.0)
        image = image * 2 - 1
        if self.transform:
            q = self.transform(q)
            a = self.transform(a)
        return image, q, a

def debug():
    train_dataloader = train_loader('clevr')
    for img, q, a in train_dataloader:
        print(img.size()) 
        print(q.size()) 
        print(a.size()) 


debug()
