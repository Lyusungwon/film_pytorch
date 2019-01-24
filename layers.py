from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    def __init__(self, vocab, embedding, hidden, num_layer=1, dropout=0, pretrained_weight=None):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=0)
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
            # self.embedding.weight.require_grad = False
        self.gru = nn.GRU(embedding, hidden, num_layers=num_layer, dropout=dropout, bidirectional=False)
        # self.gru.flatten_parameters()

    def forward(self, question, question_length):
        embedded = self.embedding(question)
        packed_embedded = pack_padded_sequence(embedded, question_length, batch_first=True)
        output, h_n = self.gru(packed_embedded)
        return output, h_n.squeeze(0)


class Conv(nn.Module):
    def __init__(self, filter, kernel, stride, layer, batchnorm):
        super(Conv, self).__init__()
        prev_filter = 3
        net = nn.ModuleList([])
        for _ in range(layer):
            net.append(nn.Conv2d(prev_filter, filter, kernel, stride, (kernel - 1)//2, bias=not batchnorm))
            if batchnorm:
                net.append(nn.BatchNorm2d(filter))
            net.append(nn.ReLU(inplace=True))
            prev_filter = filter
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


class MLP(nn.Module):
    def __init__(self, input, hidden, output, layer, dropout=None):
        super(MLP, self).__init__()
        layers = [input] + [hidden for _ in range(layer)] + [output]
        net = []
        for n, (inp, outp) in enumerate(zip(layers, layers[1:])):
            net.append(nn.Linear(inp, outp))
            net.append(nn.ReLU(inplace=True))
            if dropout and n == layer - 1:
                # net.insert(-3, nn.Dropout(dropout))
                net.append(nn.Dropout(dropout))
        net = nn.ModuleList(net[:-1])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
