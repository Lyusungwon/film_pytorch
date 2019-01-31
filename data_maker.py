import os
from collections import defaultdict, Counter
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

clevr_q_dict = {'count': 'count',
                'equal_size': 'compare_attribute',
                'equal_integer': 'compare_integer',
                'equal_shape': 'compare_attribute',
                'equal_color': 'compare_attribute',
                'equal_material': 'compare_attribute',
                'exist': 'exist',
                'less_than': 'compare_integer',
                'greater_than': 'compare_integer',
                'query_size': 'query_attribute',
                'query_shape': 'query_attribute',
                'query_material': 'query_attribute',
                'query_color': 'query_attribute'}

def make_questions(data_dir, dataset, top_k=None, multi_label=False):
    print(f"Start making {dataset} data pickle")
    if top_k and (dataset == 'vqa2' or dataset == 'vqa1'):
        answer_corpus = list()
        for mode in modes:
            annotation_file = os.path.join(data_dir, dataset, 'v2_mscoco_{}2014_annotations.json'.format(mode))
            with open(annotation_file) as f:
                annotations = json.load(f)["annotations"]
            for q_obj in annotations:
                if not multi_label:
                    answer_word = q_obj["multiple_choice_answer"]
                    answer_corpus.append(answer_word)
                else:
                    answers = q_obj["answers"]
                    for answer in answers:
                        answer_corpus.append(answer["answer"])

        top_k_words = set([i for (i, j) in Counter(answer_corpus).most_common(top_k)])
    query = 'type' if dataset == 'sample' else 'function'
    q_corpus = set()
    a_corpus = set()
    qt_corpus = set()
    qa_list = defaultdict(list)
    for mode in modes:
        if dataset == 'clevr' or dataset == 'sample':
            question_file = os.path.join(data_dir, dataset, 'questions', 'CLEVR_{}_questions.json'.format(mode))
            with open(question_file) as f:
                questions = json.load(f)['questions']
            for question in questions:
                image_dir = question['image_filename']
                image_id = int(image_dir.split('.')[0].split('_')[-1])
                q_text = question['question'].lower()
                q_text = re.sub(";", " ;", q_text)
                q_words = re.sub("[^;A-Za-z ]+", "", q_text).split(' ')
                q_corpus.update(q_words)
                a_word = [str(question['answer']).lower().strip()]
                a_corpus.update(a_word)
                q_type = clevr_q_dict[question['program'][-1][query]]
                qt_corpus.add(q_type)
                qa_list[mode].append((image_dir, image_id, q_words, a_word, q_type))
        elif dataset == 'vqa2' or dataset == 'vqa1':
            question_list = {}
            question_file = os.path.join(data_dir, dataset, f'v2_OpenEnded_mscoco_{mode}2014_questions.json')
            with open(question_file) as f:
                questions = json.load(f)["questions"]
            for question in questions:
                question_list[question['question_id']] = question['question']
            annotation_file = os.path.join(data_dir, dataset, 'v2_mscoco_{}2014_annotations.json'.format(mode))
            with open(annotation_file) as f:
                annotations = json.load(f)["annotations"]
            for q_obj in annotations:
                if not multi_label:
                    answer_word = [q_obj["multiple_choice_answer"]]
                    if top_k and answer_word[0] not in top_k_words:
                        continue
                else:
                    answer_words = q_obj["answers"]
                    answer_word = list()
                    for word in answer_words:
                        if top_k and word["answer"] in top_k_words:
                            answer_word.append(word["answer"])
                    if top_k and len(answer_word) == 0:
                        continue
                image_id = q_obj['image_id']
                image_dir = f'COCO_{mode}2014_{str(image_id).zfill(12)}.jpg'
                question_text = question_list[q_obj['question_id']]
                question_words = re.sub('[^0-9A-Za-z ]+', "", question_text).lower().split(' ')
                answer_type = q_obj["answer_type"]
                q_corpus.update(question_words)
                a_corpus.update(answer_word)
                qt_corpus.add(answer_type)
                qa_list[mode].append((image_dir, image_id, question_words, answer_word, answer_type))
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
    print(f"The number of question corpus: {len(q_corpus)}")
    print(f"The number of answers corpus: {len(a_corpus)}")
    print(f"The number of question types corpus: {len(qt_corpus)}")
    data_dict = {'word_to_idx': word_to_idx,
                 'idx_to_word': idx_to_word,
                 'answer_word_to_idx': answer_word_to_idx,
                 'answer_idx_to_word': answer_idx_to_word,
                 'question_type_to_idx': question_type_to_idx,
                 'idx_to_question_type': idx_to_question_type
                 }
    with open(os.path.join(data_dir, dataset, f'data_dict_{top_k}_{multi_label}.pkl'), 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'data_dict_{top_k}_{multi_label}.pkl saved')
    print(f"Start making {dataset} question data")
    for mode in modes:
        with h5py.File(os.path.join(data_dir, dataset, f'questions_{mode}_{top_k}_{multi_label}.h5'), 'w') as f:
            q_dset = None
            for n, (image_dir, image_id, q_word_list, answer_word, q_type) in enumerate(qa_list[mode]):
                if q_dset is None:
                    N = len(qa_list[mode])
                    dt = h5py.special_dtype(vlen=np.dtype('int32'))
                    q_dset = f.create_dataset('questions', (N,), dtype=dt)
                    a_dset = f.create_dataset('answers', (N,), dtype=dt)
                    qt_dset = f.create_dataset('question_types', (N, ), dtype='int32')
                    ii_dset = f.create_dataset('image_ids', (N,), dtype='int32')
                q = [word_to_idx[word] for word in q_word_list]
                q.append(1)
                q_dset[n] = q
                a_dset[n] = [answer_word_to_idx[word] for word in answer_word]
                ii_dset[n] = image_id
                qt_dset[n] = question_type_to_idx[q_type]
        print(f"questions_{mode}_{top_k}_{multi_label}.h5' saved")


def make_images(data_dir, dataset, size, batch_size=128, max_images=None):
    print(f"Start making {dataset} image pickle")
    model_name = 'resnet152' if dataset == 'vqa2' else 'resnet101'
    image_type = 'jpg' if dataset == 'vqa2' else 'png'
    stage = 3 if size[0] <=300 else 4
    model = build_model(model_name, stage)
    img_size = size
    idx_dict = dict()
    for mode in modes:
        img_dir = f'{mode}2014' if dataset == 'vqa2' else f'images/{mode}'
        input_paths = []
        idx_set = set()
        input_image_dir = os.path.join(data_dir, dataset, img_dir)
        idx_dict[f'{mode}'] = dict()
        for n, fn in enumerate(sorted(os.listdir(input_image_dir))):
            if not fn.endswith(image_type): continue
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            idx_dict[f'{mode}'][idx] = n
            input_paths.append((os.path.join(input_image_dir, fn), n))
            idx_set.add(idx)
        input_paths.sort(key=lambda x: x[1])
        assert len(idx_set) == len(input_paths)
        # assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
        if max_images is not None:
            input_paths = input_paths[:max_images]
        print(input_paths[0])
        print(input_paths[-1])
        with h5py.File(os.path.join(data_dir, dataset, f'images_{mode}_{str(size[0])}.h5'), 'w') as f:
            feat_dset = None
            i0 = 0
            cur_batch = []
            for i, (path, idx) in enumerate(input_paths):
                img = imread(path, mode='RGB')
                img = imresize(img, img_size, interp='bicubic')
                img = img.transpose(2, 0, 1)[None]
                cur_batch.append(img)
                if len(cur_batch) == batch_size:
                    feats = run_batch(cur_batch, model, dataset)
                    if feat_dset is None:
                        N = len(input_paths)
                        _, C, H, W = feats.shape
                        feat_dset = f.create_dataset('images', (N, C, H, W),
                                                     dtype=np.float32)
                        print(N, C, H, W)
                    i1 = i0 + len(cur_batch)
                    feat_dset[i0:i1] = feats
                    i0 = i1
                    print('Processed %d / %d images' % (i1, len(input_paths)))
                    cur_batch = []
            if len(cur_batch) > 0:
                feats = run_batch(cur_batch, model, dataset)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                print('Processed %d / %d images' % (i1, len(input_paths)))
        print(f"images saved in {os.path.join(data_dir, dataset, f'image_{mode}_{str(size[0])}.h5')}")
        with open(os.path.join(data_dir, dataset, 'idx_dict.pkl'), 'wb') as file:
            pickle.dump(idx_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('idx_dict.pkl saved')


def build_model(model, stage=4):
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


def run_batch(cur_batch, model, dataset):
    if dataset == 'clevr':
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    else:
        mean = np.array([0, 0, 0]).reshape(1, 3, 1, 1)
        std = np.array([1.0, 1.0, 1.0]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    with torch.no_grad():
        image_batch = torch.FloatTensor(image_batch).cuda()
        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()
    return feats


if __name__ =='__main__':
    data_directory = os.path.join(home, 'data')
    make_questions(data_directory, 'vqa2', 1000, True)
    # make_images(data_directory, 'sample', (448, 448), 5, 100)
    # make_questions(data_directory, 'sample')
    # make_images(data_directory, 'sample', (224, 224), 5, 100)


