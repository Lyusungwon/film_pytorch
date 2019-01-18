import os
from collections import defaultdict
import numpy as np
import json
import re
import pickle
import torch
from pathlib import Path
import h5py
from scipy.misc import imread, imresize

home = str(Path.home())

modes = ['train', 'val']


def make_clevr_questions(data_dir):
    print("Start making CLEVR data pickle")
    query = 'type' if 'sample' in data_dir else 'function'
    q_corpus = set()
    a_corpus = set()
    qt_corpus = set()
    q_list = dict()
    qa_list = defaultdict(list)
    for mode in modes:
        qf_dir = os.path.join(data_dir, 'questions', 'CLEVR_{}_questions.json'.format(mode))
        with open(qf_dir) as f:
            q_list[mode] = json.load(f)['questions']
        for q_obj in q_list[mode]:
            img_f = q_obj['image_filename']
            q_text = q_obj['question'].lower()
            q_words = re.sub('[^;A-Za-z ]+', "", q_text).split(' ')
            q_corpus.update(q_words)
            a_text = str(q_obj['answer']).lower().strip()
            a_corpus.add(a_text)
            q_type = q_obj['program'][-1][query]
            qt_corpus.add(q_type)
            qa_list[mode].append((img_f, q_words, a_text, q_type))
    word_to_idx = {"<pad>": 0, "<eos>": 1}
    idx_to_word = {0: "<pad>", 1: "<eos>"}
    answer_word_to_idx = dict()
    answer_idx_to_word = dict()
    question_type_to_idx = dict()
    idx_to_question_type = dict()
    for word in sorted(list(q_corpus)):
        word_to_idx[word] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = word
    for word in sorted(list(a_corpus)):
        answer_word_to_idx[word] = len(answer_word_to_idx)
        answer_idx_to_word[len(answer_idx_to_word)] = word
    for question_type in sorted(list(qt_corpus)):
        question_type_to_idx[question_type] = len(question_type_to_idx)
        idx_to_question_type[len(idx_to_question_type)] = question_type
    data_dict = {'word_to_idx': word_to_idx,
                 'idx_to_word': idx_to_word,
                 'answer_word_to_idx': answer_word_to_idx,
                 'answer_idx_to_word': answer_idx_to_word,
                 'question_type_to_idx': question_type_to_idx,
                 'idx_to_question_type': idx_to_question_type
                 }
    with open(os.path.join(data_dir, 'data_dict.pkl'), 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('data_dict.pkl saved')
    print("Start making CLEVR question data")
    qa_idx_data = defaultdict(list)
    for mode in modes:
        with h5py.File(os.path.join(data_dir, f'questions_{mode}.h5'), 'w') as f:
            q_dset = None
            for n, (img_file, q_word_list, answer_word, q_type) in enumerate(qa_list[mode]):
                if q_dset is None:
                    N = len(qa_list[mode])
                    dt = h5py.special_dtype(vlen=np.dtype('int32'))
                    q_dset = f.create_dataset('questions', (N,), dtype=dt)
                q = [word_to_idx[word] for word in q_word_list]
                q.append(1)
                q_dset[n] = q
                a = answer_word_to_idx[answer_word]
                q_t = question_type_to_idx[q_type]
                qa_idx_data[mode].append((img_file, a, q_t))
        with open(os.path.join(data_dir, 'data_{}.pkl'.format(mode)), 'wb') as file:
            pickle.dump(qa_idx_data[mode], file, protocol=pickle.HIGHEST_PROTOCOL)
        print('data_{}.pkl saved'.format(mode))


def make_clevr_images(data_dir, size, batch_size=12):
    print("Start making CLEVR image pickle")
    model = build_model()
    img_size = size
    for mode in modes:
        input_paths = []
        idx_set = set()
        input_image_dir = os.path.join(data_dir, 'images', f'{mode}')
        for fn in os.listdir(input_image_dir):
            if not fn.endswith('.png'): continue
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            input_paths.append((os.path.join(input_image_dir, fn), idx))
            idx_set.add(idx)
        input_paths.sort(key=lambda x: x[1])
        assert len(idx_set) == len(input_paths)
        assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
        # if args.max_images is not None:
        #     input_paths = input_paths[:args.max_images]
        print(input_paths[0])
        print(input_paths[-1])
        with h5py.File(os.path.join(data_dir, f'images_{mode}_{str(size[0])}.h5'), 'w') as f:
            feat_dset = None
            i0 = 0
            cur_batch = []
            for i, (path, idx) in enumerate(input_paths):
                img = imread(path, mode='RGB')
                img = imresize(img, img_size, interp='bicubic')
                img = img.transpose(2, 0, 1)[None]
                cur_batch.append(img)
                if len(cur_batch) == batch_size:
                    feats = run_batch(cur_batch, model)
                    if feat_dset is None:
                        N = len(input_paths)
                        _, C, H, W = feats.shape
                        feat_dset = f.create_dataset('images', (N, C, H, W),
                                                     dtype=np.float32)
                    i1 = i0 + len(cur_batch)
                    feat_dset[i0:i1] = feats
                    i0 = i1
                    print('Processed %d / %d images' % (i1, len(input_paths)))
                    cur_batch = []
            if len(cur_batch) > 0:
                feats = run_batch(cur_batch, model)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                print('Processed %d / %d images' % (i1, len(input_paths)))

        print(f"images saved in {os.path.join(data_dir, f'image_{mode}_{str(size[0])}.h5')}")


def build_model(model='resnet101', stage=3):
    import torchvision.models
    if not hasattr(torchvision.models, model):
        raise ValueError('Invalid model "%s"' % model)
    if not 'resnet' in model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, model)(pretrained=True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    with torch.no_grad():
        image_batch = torch.FloatTensor(image_batch).cuda()
        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()
    return feats


if __name__ =='__main__':
    data_directory = os.path.join(home, 'data/clevr')
    make_clevr(data_directory)

# question_type_dict = {'exist': 10,
#                     'count': 20,
#                     'equal_integer': 30,
#                     'less_than': 31,
#                     'greater_than': 32,
#                     'query_size': 40,
#                     'query_color': 41,
#                     'query_material': 42,
#                     'query_shape': 43,
#                     'equal_size': 50,
#                     'equal_color': 51,
#                     'equal_material': 52,
#                     'equal_shape': 53
#                     }

