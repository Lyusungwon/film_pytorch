import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle
from PIL import Image
from pathlib import Path
from data_maker import make_questions, make_images
from utils import is_file_exist, to_onehot
import h5py

home = str(Path.home())


def collate_text(list_inputs):
    list_inputs.sort(key=lambda x: len(x[1]), reverse=True)
    images = []
    questions = []
    q_length = []
    answers = []
    question_types = []
    for i, q, a, types in list_inputs:
        images.append(i)
        questions.append(q)
        q_length.append(len(q))
        answers.append(a)
        question_types.append(types)
    images = torch.cat(images, 0)
    padded_questions = pad_sequence(questions, batch_first=True)
    q_length = torch.Tensor(q_length).to(torch.long)
    answers = torch.cat(answers, 0)
    question_types = torch.cat(question_types, 0)
    return images, (padded_questions, q_length), answers, question_types


def load_dataloader(data_directory, dataset, is_train=True, batch_size=128, data_config=[224, 224, 0, True, 0, False]):
    input_h, input_w, cpu_num, cv_pretrained, top_k, multi_label = data_config
    if cv_pretrained:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((input_h, input_w)), transforms.ToTensor()])
    dataloader = DataLoader(
        VQA(data_directory, dataset, train=is_train, cv_pretrained=cv_pretrained, transform=transform, size=(input_h, input_w), top_k=top_k, multi_label=multi_label),
        batch_size=batch_size, shuffle=True,
        num_workers=cpu_num, pin_memory=True,
        collate_fn=collate_text)
    return dataloader


class VQA(Dataset):
    """VQA dataset."""
    def __init__(self, data_dir, dataset, train=True, cv_pretrained=True, transform=None, size=(224,224), top_k=0, multi_label=False):
        self.dataset = dataset
        self.mode = 'train' if train else 'val'
        self.cv_pretrained = cv_pretrained
        self.transform = transform
        self.multi_label = multi_label
        self.data_file = os.path.join(data_dir, dataset, f'data_dict_{top_k}_{multi_label}.pkl')
        self.question_file = os.path.join(data_dir, dataset, f'questions_{self.mode}_{top_k}_{multi_label}.h5')
        if self.cv_pretrained:
            self.image_dir = os.path.join(data_dir, dataset, f'images_{self.mode}_{str(size[0])}.h5')
            self.idx_dict_file = os.path.join(data_dir, dataset, 'idx_dict.pkl')
        else:
            if dataset == 'clevr' or dataset == 'sample':
                self.image_dir = os.path.join(data_dir, dataset, 'images', f'{self.mode}')
            elif dataset == 'vqa2':
                self.image_dir = os.path.join(data_dir, dataset, f'{self.mode}2014')
        if not is_file_exist(self.question_file):
            make_questions(data_dir, dataset, top_k, multi_label)
        if cv_pretrained:
            if not is_file_exist(self.image_dir):
                make_images(data_dir, dataset, size)
        self.load_data()


    def load_data(self):
        if self.multi_label:
            print(f"Start loading {self.data_file}")
            with open(self.data_file, 'rb') as file:
                data_dict = pickle.load(file)
                self.a_size = len(data_dict['answer_word_to_idx'])
        if self.cv_pretrained:
            print(f"Start loading {self.idx_dict_file}")
            with open(self.idx_dict_file, 'rb') as file:
                self.idx_dict = pickle.load(file)[self.mode]

    def __len__(self):
        return h5py.File(self.question_file, 'r', swmr=True)['questions'].shape[0]

    def __getitem__(self, idx):
        question_file = h5py.File(self.question_file, 'r', swmr=True)
        q = question_file['questions'][idx]
        a = question_file['answers'][idx]
        q_t = question_file['question_types'][idx]
        ii = question_file['image_ids'][idx]
        if self.cv_pretrained:
            image = h5py.File(self.image_dir, 'r', swmr=True)['images'][self.idx_dict[ii]]
            image = torch.from_numpy(image).unsqueeze(0)
        else:
            image_file = f'COCO_{self.mode}2014_{str(ii).zfill(12)}.jpg' if 'vqa' in self.dataset else f'CLEVR_{self.mode}_{str(ii).zfill(6)}.png'
            if self.dataset == 'sample':
                image_file = f'CLEVR_new_{str(ii).zfill(6)}.png'
            image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
            if self.transform:
                image = self.transform(image).unsqueeze(0)
        q = torch.from_numpy(q).to(torch.long)
        a = torch.Tensor(a).to(torch.long) if not self.multi_label else to_onehot(a, self.a_size)
        q_t = torch.Tensor([q_t]).to(torch.long)
        return image, q, a, q_t


if __name__ =='__main__':
    dataloader = load_dataloader(os.path.join(home, 'data'), 'sample', True, 2, data_config=[224, 224, 0, True, 0, True])
    for img, q, a, types in dataloader:
        print(img.size())
        print(q)
        print(a.size())
        print(types.size())
        break
