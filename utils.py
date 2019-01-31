import os
import torch
import torch.nn as nn
import time
import pickle


def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2 - t1))
        return result
    return wrap


def is_file_exist(file):
    if os.path.isfile(file):
        print(f"Data {file} exist")
        return True
    else:
        print(f"Data {file} does not exist")
        return False


def positional_encode(images):
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    x_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images


def save_checkpoint(epoch_idx, model, optimizer, args, batch_record_idx):
    log = args.log
    checkpoint = dict()
    checkpoint['model'] = model
    checkpoint['model_parameters'] = model.state_dict()
    checkpoint['optimizer_parameters'] = optimizer.state_dict()
    checkpoint['args'] = args
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch_idx'] = batch_record_idx
    save_file = os.path.join(log, 'checkpoint.pt')
    torch.save(checkpoint, save_file)
    print('Model saved in {}'.format(save_file))
    return True


def load_checkpoint(model, optimizer, log, device):
    load_file = os.path.join(log, 'checkpoint.pt')
    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint['model_parameters'])
    optimizer.load_state_dict(checkpoint['optimizer_parameters'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    epoch_idx = checkpoint['epoch']
    batch_record_idx = checkpoint['batch_idx']
    print('Model loaded from {}.'.format(load_file))
    return model, optimizer, epoch_idx, batch_record_idx


def load_pretrained_embedding(word2idx, embedding_dim):
    import torchtext
    pretrained = torchtext.vocab.GloVe(name='6B', dim=embedding_dim)
    embedding = torch.Tensor(len(word2idx), embedding_dim)
    for word, idx in word2idx.items():
        if word != "<eos>":
            embedding[idx, :] = pretrained[word].data
    embedding[word2idx["<eos>"], :] = embedding.mean(0).data
    print("Loaded pretrained embedding.")
    return embedding


def load_pretrained_conv():
    import torchvision.models as models
    model = models.resnet101(pretrained=True)
    feature_extractor = list(model.children())[:-1]
    for part in feature_extractor:
        for param in part.parameters():
            param.requires_grad = False
    # if output_channel:
    #     feature_extractor.append(nn.Conv2d(1024, output_channel, 1, 1))
    feature_extractor = nn.Sequential(*feature_extractor)
    print("Loaded pretrained feature extraction model.")
    return feature_extractor


def load_dict(args):
    dict_file = os.path.join(args.data_directory, args.dataset, f'data_dict_{args.top_k}_{args.multi_label}.pkl')
    with open(dict_file, 'rb') as file:
        data_dict = pickle.load(file)
    args.word_to_idx = data_dict['word_to_idx']
    args.idx_to_word = data_dict['idx_to_word']
    args.answer_word_to_idx = data_dict['answer_word_to_idx']
    args.answer_idx_to_word = data_dict['answer_idx_to_word']
    args.question_type_to_idx = data_dict['question_type_to_idx']
    args.idx_to_question_type = data_dict['idx_to_question_type']
    args.q_size = len(args.word_to_idx)
    args.a_size = len(args.answer_word_to_idx)
    args.qt_size = len(args.question_type_to_idx)
    return args


def to_onehot(a, a_size):
    onehot = torch.zeros(len(a), a_size)
    divide = 3.0 if len(a) > 2 else 1.0
    onehot[[i for i in range(len(a))], a] = 1.0
    onehot = torch.min(onehot.sum(0) / divide, torch.ones(1)).unsqueeze(0)
    # onehot = onehot.sum(0).unsqueeze(0) / float(len(a))
    return onehot

# def make_model(args):
#     if args.model == 'film':
#         model = Film(args)
#     elif args.model == 'san':
#         model = San(args)
#     elif args.model == 'rn':
#         model = RelationalNetwork(args)
#     return model
# if __name__ == '__main__':
#     dict_file = os.path.join('/home/sungwon/data', 'vqa2', 'data_dict_0.pkl')
#     with open(dict_file, 'rb') as file:
#         data_dict = pickle.load(file)
#     idx_to_question_type = data_dict['idx_to_question_type']
#     print(idx_to_question_type)
