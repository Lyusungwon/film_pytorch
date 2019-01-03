import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import positional_encode


class Text_encoder(nn.Module):
    def __init__(self, vocab, embedding, hidden, num_layer=1, dropout=0):
        super(Text_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=0)
        self.gru = nn.GRU(embedding, hidden, num_layers=num_layer, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, x):
        embedded = self.embedding(x[0])
        packed_embedded = pack_padded_sequence(embedded, x[1])
        output, h_n = self.gru(packed_embedded)
        return h_n.squeeze(0)

# class Text_embedding(nn.Module):
#     def __init__(self, color_size, question_size, embedding_size):
#         super(Text_embedding, self).__init__()
#         self.color_embedding = nn.Embedding(color_size, embedding_size, padding_idx=None)
#         self.question_embedding = nn.Embedding(question_size, embedding_size, padding_idx=None)
#
#     def forward(self, x):
#         c_embedded = self.color_embedding(x[:, 0])
#         q_embedded = self.question_embedding(x[:, 1])
#         text_embedded = torch.cat([c_embedded, q_embedded], 1)
#         return text_embedded


class Conv(nn.Module):
    def __init__(self, filter, kernel, stride, layer, batchnorm):
        super(Conv, self).__init__()
        prev_filter = 3
        net = nn.ModuleList([])
        for _ in range(layer):
            net.append(nn.Conv2d(prev_filter, filter, kernel, stride, (kernel - 1)//2))
            if batchnorm:
                net.append(nn.BatchNorm2d(filter))
            net.append(nn.ReLU(inplace=True))
            prev_filter = filter
        self.net = nn.Sequential(*net)
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, filter, kernel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter + 2, filter, 1, 1, 0)
        self.conv2 = nn.Conv2d(filter, filter, kernel, 1, (kernel - 1)//2)
        self.batch_norm = nn.BatchNorm2d(filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, betagamma):
        x = positional_encode(x)
        x = self.relu(self.conv1(x))
        residual = x
        beta = betagamma[:, 0].unsqueeze(2).unsqueeze(3).expand_as(x)
        gamma = betagamma[:, 1].unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.batch_norm(self.conv2(x))
        x = self.relu(x * beta + gamma)
        x = x + residual
        return x


class Classifier(nn.Module):
    def __init__(self, filter, last_filter, hidden, last, layer):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(filter + 2, last_filter, 1, 1, 0)
        # self.pool = nn.MaxPool2d((input_h, input_w))
        self.mlp = MLP(last_filter, hidden, last, layer)

    def forward(self, x):
        x = positional_encode(x)
        x = self.conv(x).max(2)[0].max(2)[0]
        x = self.mlp(x)
        return x


class MLP(nn.Module):
    def __init__(self, input, hidden, output, layer):
        super(MLP, self).__init__()
        layers = [input] + [hidden for _ in range(layer)] + [output]
        net = []
        for n, (inp, outp) in enumerate(zip(layers, layers[1:])):
            net.append(nn.Linear(inp, outp))
            net.append(nn.ReLU(inplace=True))
            # if self.dropout == n + 1:
            #     net.append(nn.Dropout(self.dropout_rate))
        net = nn.ModuleList(net[:-1])
        self.net = nn.Sequential(*net)
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x
