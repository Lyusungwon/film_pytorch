import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle
from PIL import Image
from pathlib import Path
from clevr_maker import make_clevr_images, make_clevr_questions
from vqa2_maker import make_vqa2
import h5py
import numpy as np

home = str(Path.home())


def collate_clevr(list_inputs):
    list_inputs.sort(key=lambda x: len(x[1]), reverse=True)
    images = []
    questions = []
    q_length = []
    answers = []
    question_types = []
    for i, q, a, q_type in list_inputs:
        images.append(i)
        questions.append(q)
        q_length.append(len(q))
        answers.append(a)
        question_types.append(q_type)
    images = torch.cat(images, 0)
    padded_questions = pad_sequence(questions, batch_first=True)
    q_length = torch.Tensor(q_length).to(torch.long)
    answers = torch.Tensor(answers).to(torch.long)
    question_types = torch.Tensor(question_types).to(torch.long)
    return images, (padded_questions, q_length), answers, [question_types]


def collate_vqa(list_inputs):
    list_inputs.sort(key=lambda x: len(x[1]), reverse=True)
    images = torch.Tensor()
    questions = []
    q_length = []
    answers = []
    question_types = []
    answer_types = []
    for i, q, a, q_type, a_type in list_inputs:
        images = torch.cat([images, i.unsqueeze(0)], 0)
        questions.append(q)
        q_length.append(len(q))
        answers.append(a)
        question_types.append(q_type)
        answer_types.append(a_type)
    answers = torch.cat(answers, 0)
    question_types = torch.cat(question_types, 0)
    answer_types = torch.cat(answer_types, 0)
    padded_questions = pad_sequence(questions, batch_first=True)
    q_length = torch.Tensor(q_length).to(torch.long)
    return images, (padded_questions, q_length), answers, [question_types, answer_types]


def load_dataloader(data, data_directory, is_train=True, batch_size=128, data_config=[224, 224, 0, True]):
    input_h, input_w, cpu_num, cv_pretrained = data_config
    if cv_pretrained:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((input_h, input_w)), transforms.ToTensor()])
    if data == 'clevr' or data == 'sample':
        dataloader = DataLoader(
            Clevr(os.path.join(data_directory, data), train=is_train, cv_pretrained=cv_pretrained,
            transform=transform, size=(input_h, input_w)),
            batch_size=batch_size, shuffle=True,
            num_workers=cpu_num, pin_memory=True,
            collate_fn=collate_clevr)
    elif data == 'vqa2':
        dataloader = DataLoader(
            VQA2(os.path.join(data_directory, data), train=is_train, cv_pretrained=cv_pretrained,
            transform=transform),
            batch_size=batch_size, shuffle=True,
            num_workers=cpu_num, pin_memory=True,
            collate_fn=collate_vqa)
    return dataloader


class Clevr(Dataset):
    """Clevr dataset."""
    def __init__(self, data_dir, train=True, cv_pretrained=True, transform=None, size=(224,224)):
        self.mode = 'train' if train else 'val'
        self.cv_pretrained = cv_pretrained
        self.transform = transform
        if self.cv_pretrained:
            self.img_dir = os.path.join(data_dir, f'images_{self.mode}_{str(size[0])}.h5')
        else:
            self.img_dir = os.path.join(data_dir, 'images', f'{self.mode}')
        self.question_file = os.path.join(data_dir, f'questions_{self.mode}.h5')
        self.data_file = os.path.join(data_dir, 'data_{}.pkl'.format(self.mode))
        # self.dict_file = os.path.join(data_dir, 'data_dict.pkl')
        if not self.is_file_exits(self.question_file):
            make_clevr_questions(data_dir)
        if cv_pretrained:
            if not self.is_file_exits(self.img_dir):
                make_clevr_images(data_dir, size)
        self.load_data()

    def is_file_exits(self, file):
        if os.path.isfile(file):
            print(f"Data {file} exist")
            return True
        else:
            print(f"Data {file} does not exist")
            return False

    def load_data(self):
        print("Start loading {}".format(self.data_file))
        if self.cv_pretrained:
            self.images = h5py.File(self.img_dir, 'r')['images']
            print(self.images.shape)
        self.questions = h5py.File(self.question_file, 'r')['questions']
        print(self.questions.shape)
        with open(self.data_file, 'rb') as file:
            self.data = pickle.load(file)
            print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, a, q_t = self.data[idx]
        q = self.questions[idx]
        if not self.cv_pretrained:
            image = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')
            if self.transform:
                image = self.transform(image).unsqueeze(0)
        else:
            image_idx = int(img_file.split('.')[0].split('_')[-1])
            image = self.images[image_idx]
            image = torch.from_numpy(image).unsqueeze(0)
        q = torch.from_numpy(q).to(torch.long)
        return image, q, a, q_t


class VQA2(Dataset):
    """VQA2.0 dataset."""
    def __init__(self, data_dir, train=True, reduced_data=False, transform=None):
        self.mode = 'train' if train else 'val'
        self.reduced_data = reduced_data
        self.transform = transform
        self.reduced_data = reduced_data
        if not reduced_data:
            self.img_dir = os.path.join(data_dir, f'{self.mode}2014')
        else:
            self.img_dir = os.path.join(data_dir, f'{self.mode}_reduced_images.pkl')
        self.data_file = os.path.join(data_dir, 'data_{}.pkl'.format(self.mode))
        self.dict_file = os.path.join(data_dir, 'data_dict.pkl')
        if not self.is_file_exits(self.data_file):
            make_vqa2(data_dir)
        self.load_data()

    def is_file_exits(self, file):
        if os.path.isfile(file):
            print(f"Data {file} exist")
            return True
        else:
            print(f"Data {file} does not exist")
            return False

    def load_data(self):
        print("Start loading {}".format(self.data_file))
        with open(self.data_file, 'rb') as file:
            self.data = pickle.load(file)
        if self.reduced_data:
            if not self.is_file_exits(self.img_dir):
                raise
            with open(self.img_dir, 'rb') as file:
                self.images = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_idx, q, a, q_t, a_t = self.data[idx]
        if not self.reduced_data:
            image = Image.open(os.path.join(self.img_dir, 'COCO_{}2014_{}.jpg'.format(self.mode, str(image_idx).zfill(12)))).convert('RGB')
            print(image.type())
            if self.transform:
                image = self.transform(image)
            image = np.array(image)
        else:
            image = self.images[idx]
            image = torch.from_numpy(image).unsqueeze(0)
        q = torch.from_numpy(q).to(torch.long)
        return image, q, a, q_t, a_t


if __name__ =='__main__':
    dataloader = load_dataloader('sample', os.path.join(home, 'data'), True, 2, data_config=[224, 224, 0, True])
    for img, q, a, types in dataloader:
        print(img.size())
        print(q)
        print(a)
        print(types)
