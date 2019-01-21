import os
import torch
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


def positional_encode(images):
    device = images.get_device()
    n, c, h, w = images.size()
    x_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images


def baseline_encode(images, questions):
    device = images.get_device()
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-h / 2, h / 2, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-w / 2, w / 2, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    questions = questions.unsqueeze(2).unsqueeze(3).expand(n, hd, h, w)
    images = torch.cat([images, x_coordinate, y_coordinate, questions], 1).view(n, -1, o).transpose(1, 2)
    return images


def rn_encode(images, questions):
    device = images.get_device()
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-1, 1, h).view(1, h, 1, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    images = images.view(n, c, o).transpose(1, 2)
    images = torch.cat([images, x_coordinate, y_coordinate], 2)
    images1 = images.unsqueeze(1).expand(n, o, o, c + 2).contiguous()
    images2 = images.unsqueeze(2).expand(n, o, o, c + 2).contiguous()
    questions = questions.unsqueeze(1).unsqueeze(2).expand(n, o, o, hd)
    # pairs = torch.cat([images1, images2, questions], 3).view(n, o**2, -1)
    pairs = torch.cat([images1, images2, questions], 3)
    return pairs


def lower_sum(relations):
    device = relations.get_device()
    n, h, w, l = relations.size()
    mask = torch.ones([h, w]).tril().view(1, h, w, 1).to(device, dtype=torch.uint8)
    relations = torch.masked_select(relations, mask).view(n, -1, l)
    return relations.sum(1)


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


# def load_pretrained_conv(output_channel=None):
#     import torchvision.models as models
#     model = models.resnet101(pretrained=True)
#     feature_extractor = list(model.children())[:-3]
#     for part in feature_extractor:
#         for param in part.parameters():
#             param.requires_grad = False
#     if output_channel:
#         feature_extractor.append(nn.Conv2d(1024, output_channel, 1, 1))
#     feature_extractor = nn.Sequential(*feature_extractor)
#     print("Loaded pretrained feature extraction model.")
#     return feature_extractor


def load_dict(args):
    dict_file = os.path.join(args.data_directory, args.dataset, 'data_dict.pkl')
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


# def make_model(args):
#     if args.model == 'film':
#         model = Film(args)
#     elif args.model == 'san':
#         model = San(args)
#     elif args.model == 'rn':
#         model = RelationalNetwork(args)
#     return model
