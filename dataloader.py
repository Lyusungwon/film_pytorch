import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle
from PIL import Image
from pathlib import Path
from clevr_maker import make_clevr
from vqa2_maker import make_vqa2
home = str(Path.home())


def collate_clevr(list_inputs):
    list_inputs.sort(key=lambda x: len(x[1]), reverse=True)
    images = torch.Tensor()
    questions = []
    q_length = []
    answers = torch.Tensor().to(torch.long)
    question_types = torch.Tensor().to(torch.long)
    for i, q, a, q_type in list_inputs:
        images = torch.cat([images, i.unsqueeze(0)], 0)
        questions.append(q)
        q_length.append(len(q))
        answers = torch.cat([answers, a], 0)
        question_types = torch.cat([question_types, q_type], 0)
    padded_questions = pad_sequence(questions, batch_first=True)
    q_length = torch.Tensor(q_length).to(torch.long)
    return images, (padded_questions, q_length), answers, [question_types]


def collate_vqa(list_inputs):
    list_inputs.sort(key=lambda x: len(x[1]), reverse=True)
    images = torch.Tensor()
    questions = []
    q_length = []
    answers = torch.Tensor().to(torch.long)
    question_types = torch.Tensor().to(torch.long)
    answer_types = torch.Tensor().to(torch.long)
    for i, q, a, q_type, a_type in list_inputs:
        images = torch.cat([images, i.unsqueeze(0)], 0)
        questions.append(q)
        q_length.append(len(q))
        answers = torch.cat([answers, a], 0)
        question_types = torch.cat([question_types, q_type], 0)
        answer_types = torch.cat([answer_types, a_type], 0)
    padded_questions = pad_sequence(questions, batch_first=True)
    q_length = torch.Tensor(q_length).to(torch.long)
    return images, (padded_questions, q_length), answers, [question_types, answer_types]


def load_dataloader(data, data_directory, is_train=True, batch_size=128, data_config=[224, 224, 0]):
    input_h, input_w, cpu_num = data_config
    if data == 'clevr' or data == 'sample':
        dataloader = DataLoader(
            Clevr(os.path.join(data_directory, data), train=is_train,
            transform=transforms.Compose([transforms.Resize((input_h, input_w)), transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True,
            num_workers=cpu_num, pin_memory=True,
            collate_fn=collate_clevr)
        return dataloader
    elif data == 'vqa2':
        dataloader = DataLoader(
            VQA2(os.path.join(data_directory, data), train=is_train,
            transform=transforms.Compose([transforms.Resize((input_h, input_w)), transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True,
            num_workers=cpu_num, pin_memory=True,
            collate_fn=collate_vqa)
        return dataloader


class Clevr(Dataset):
    """Clevr dataset."""
    def __init__(self, data_dir, train=True, transform=None):
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.q_dir = os.path.join(data_dir, 'questions', 'CLEVR_{}_questions.json'.format(self.mode))
        self.img_dir = os.path.join(data_dir, 'images', '{}'.format(self.mode))
        self.data_file = os.path.join(data_dir, 'data_{}.pkl'.format(self.mode))
        self.dict_file = os.path.join(data_dir, 'data_dict.pkl')
        if not self.is_file_exits():
            make_clevr(data_dir)
        self.load_data()

    def is_file_exits(self):
        if os.path.isfile(self.data_file):
            print("Data {} exist".format(self.data_file))
            return True
        else:
            print("Data {} does not exist".format(self.data_file))
            return False

    def load_data(self):
        print("Start loading {}".format(self.data_file))
        with open(self.data_file, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, q, a, q_t = self.data[idx]
        image = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, q, a, q_t


class VQA2(Dataset):
    """VQA2.0 dataset."""
    def __init__(self, data_dir, train=True, transform=None):
        self.mode = 'train' if train else 'val'
        self.transform = transform
        # self.q_dir = os.path.join(data_dir, 'questions', 'CLEVR_{}_questions.json'.format(self.mode))
        self.img_dir = os.path.join(data_dir, '{}2014'.format(self.mode))
        self.data_file = os.path.join(data_dir, 'data_{}.pkl'.format(self.mode))
        self.dict_file = os.path.join(data_dir, 'data_dict.pkl')
        if not self.is_file_exits():
            make_vqa2(data_dir)
        self.load_data()

    def is_file_exits(self):
        if os.path.isfile(self.data_file):
            print("Data {} exist".format(self.data_file))
            return True
        else:
            print("Data {} does not exist".format(self.data_file))
            return False

    def load_data(self):
        print("Start loading {}".format(self.data_file))
        with open(self.data_file, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_idx, q, a, q_t, a_t = self.data[idx]
        image = Image.open(os.path.join(self.img_dir, 'COCO_{}2014_{}.jpg'.format(self.mode, str(image_idx).zfill(12)))).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, q, a, q_t, a_t


if __name__ =='__main__':
    dataloader = load_dataloader('vqa2', os.path.join(home, 'data'), True, 2)
    for img, q, a, types in dataloader:
        print(img.size())
        print(q)
        print(a)
        print(types)
        break
