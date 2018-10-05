import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class MLP(nn.Module):
    def __init__(self, layers, dropout = None, dropout_rate = None, last = False):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.last = last
        net = []
        for n, (inp, outp) in enumerate(zip(layers, layers[1:])):
            net.append(nn.Linear(inp, outp))
            net.append(nn.ReLU(inplace=True))
            if self.dropout == n + 1:
               net.append(nn.Dropout(self.dropout_rate))
        if self.last:
            net = net[:-1]
        net = nn.ModuleList(net)
        self.net = nn.Sequential(*net)
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x

class Conv(nn.Module):
    def __init__(self, layer_config, channel_size, batch_norm):
        super(Conv, self).__init__()
        self.layer_config = layer_config
        self.channel_size = channel_size
        self.batch_norm = batch_norm
        prev_filter = self.channel_size
        net = nn.ModuleList([])
        for num_filter, kernel_size, stride in layer_config:
            net.append(nn.Conv2d(prev_filter, num_filter, kernel_size, stride, (kernel_size - 1)//2))
            if batch_norm:
                net.append(nn.BatchNorm2d(num_filter))
            net.append(nn.ReLU(inplace=True))
            prev_filter = num_filter
        self.net = nn.Sequential(*net)
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x

class Text_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer):
        super(Text_encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=None)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layer, bidirectional=False, batch_first = True)

    def forward(self, x):
        embedded = self.embedding(x.data)
        packed_embedded = PackedSequence(embedded, x.batch_sizes)
        output, (h_n, c_n) = self.lstm(packed_embedded)
        return h_n.squeeze(0)

class Text_embedding(nn.Module):
    def __init__(self, color_size, question_size, embedding_size):
        super(Text_embedding, self).__init__()
        self.color_embedding = nn.Embedding(color_size, embedding_size, padding_idx=None)
        self.question_embedding = nn.Embedding(question_size, embedding_size, padding_idx=None)

    def forward(self, x):
        c_embedded = self.color_embedding(x[:, 0])
        q_embedded = self.question_embedding(x[:, 1])
        text_embedded = torch.cat([c_embedded, q_embedded], 1)
        return text_embedded