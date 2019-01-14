import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
        return result
    return wrap


def positional_encode(images):
    device = images.get_device()
    n, c, h, w = images.size()
    x_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images


def save_checkpoint(epoch_idx, model, optimizer, log):
    checkpoint = dict()
    checkpoint['model_parameters'] = model.state_dict()
    checkpoint['optimizer_parameters'] = optimizer.state_dict()
    checkpoint['epoch'] = epoch_idx
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
    print('Model loaded from {}.'.format(load_file))
    return model, optimizer, epoch_idx


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


def load_pretrained_conv(output_channel):
    import torchvision.models as models
    model = models.resnet101(pretrained=True)
    feature_extractor = list(model.children())[:-3]
    for part in feature_extractor:
        for param in part.parameters():
            param.requires_grad = False
    feature_extractor.append(nn.Conv2d(1024, output_channel, 1, 1))
    feature_extractor = nn.Sequential(*feature_extractor)
    print("Loaded pretrained feature extraction model.")
    return feature_extractor
