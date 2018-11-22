import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np


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
        net = nn.ModuleList(net[:-1])
        self.net = nn.Sequential(*net)
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class Conv(nn.Module):
    def __init__(self, input_h, input_w, layer_config, channel_size, layer_norm):
        super(Conv, self).__init__()
        self.layer_config = layer_config
        self.channel_size = channel_size
        self.layer_norm = layer_norm
        self.input_h = input_h
        self.input_w = input_w
        prev_filter = self.channel_size
        net = nn.ModuleList([])
        for num_filter, kernel_size, stride in layer_config:
            net.append(nn.Conv2d(prev_filter, num_filter, kernel_size, stride, (kernel_size - 1)//2))
            if layer_norm:
                self.input_h = int(np.ceil(self.input_h / 2))
                self.input_w = int(np.ceil(self.input_w / 2))
                net.append(nn.LayerNorm([num_filter, self.input_h, self.input_w]))
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

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + 2 + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + 2 + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class Film(nn.Module):
    def __init__(self, input_h, input_w, layer_config, channel_size, layer_norm, chaining_size, hidden_size, embedding_size, output_size):
        super(Film, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.layer_config = layer_config
        self.layer_size = len(layer_config)
        self.channel_size = channel_size
        self.layer_norm = layer_norm
        self.chaining_size = chaining_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.filter_size = layer_config[0][0]
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.selector = nn.LSTM(self.filter_size + embedding_size, output_size)
        self.beta_l = nn.Linear(hidden_size, self.filter_size * self.layer_size)
        self.gamma_l = nn.Linear(hidden_size, self.filter_size * self.layer_size)
        prev_filter = channel_size
        net = nn.ModuleList([])
        for num_filter, kernel_size, stride in layer_config:
            net.append(nn.Conv2d(prev_filter, num_filter, kernel_size, stride, (kernel_size - 1)//2))
            if layer_norm:
                self.input_h = int(np.ceil(self.input_h / 2))
                self.input_w = int(np.ceil(self.input_w / 2))
                net.append(nn.LayerNorm([num_filter, self.input_h, self.input_w]))
            net.append(nn.ReLU(inplace=True))
            prev_filter = num_filter
        net.append(self.lstm)
        net.append(self.selector)
        net.append(self.beta_l)
        net.append(self.gamma_l)
        self.net = net
        print(self.net)

    def forward(self, x, q):
        qs = q.unsqueeze(0).expand((self.chaining_size, -1, -1))
        output, (hn, cn) = self.lstm(qs)
        entities = torch.zeros(0, x.size()[0], self.filter_size + self.embedding_size).to(x.get_device())
        for n in range(self.chaining_size):
            x_ = x.clone()
            betas = self.beta_l(output[n])
            gammas = self.gamma_l(output[n])
            layer = 0
            for module in self.net[:-6]:
                if isinstance(module, nn.ReLU):
                    beta = betas[:, layer*self.filter_size:(layer+1)*self.filter_size].unsqueeze(2).unsqueeze(3).expand_as(x_)
                    gamma = gammas[:, layer*self.filter_size:(layer+1)*self.filter_size].unsqueeze(2).unsqueeze(3).expand_as(x_)
                    x_ = x_ * beta + gamma
                    layer += 1
                x_ = module(x_)
            x_ = torch.cat([x_.squeeze(3).squeeze(2), q], 1)
            entities = torch.cat([entities, x_.unsqueeze(0)], 0)
        output, (hn, cn) = self.selector(entities)
        return hn[-1]
#
#
# class FeatureWiseLinearTransformation(nn.Module):
#     def __init__(self, channel_size):
#         super(Forward, self).__init__()
#         self.channel_size = channel_size
#         self.beta = torch.nn.Parameter(torch.randn(1, 1, 1, channel_size))
#         self.gamma = torch.nn.Parameter(torch.randn(1, 1, 1, channel_size))
#
#     def forward(self, x):
#         assert(x.size[-1] == self.channel_size)
#         return x * self.beta + self.gamma
