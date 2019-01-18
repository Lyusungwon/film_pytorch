from layers import *
from utils import load_pretrained_embedding, load_pretrained_conv, positional_encode


class Film(nn.Module):
    def __init__(self, args):
        super(Film, self).__init__()
        self.filters = args.cv_filter
        self.layers = args.film_res_layer
        if args.te_pretrained:
            pretrained_weight = load_pretrained_embedding(args.word2idx, args.te_embedding)
        else:
            pretrained_weight = None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, pretrained_weight)
        if args.cv_pretrained:
            self.visual_encoder = nn.Conv2d(1024, args.cv_filter, 1, 1)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.fc = nn.Linear(args.te_hidden, args.cv_filter * args.film_res_layer * 2)
        self.res_blocks = nn.ModuleList([FilmResBlock(args.cv_filter, args.film_res_kernel) for _ in range(args.film_res_layer)])
        self.classifier = FilmClassifier(args.cv_filter, args.film_cf_filter, args.film_fc_hidden, args.a_size, args.film_fc_layer)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        code = self.text_encoder(question, question_length)
        betagamma = self.fc(code).view(-1, self.layers, 2, self.filters)
        for n, block in enumerate(self.res_blocks):
            x = block(x, betagamma[:, n])
        logits = self.classifier(x)
        return logits


class FilmResBlock(nn.Module):
    def __init__(self, filter, kernel):
        super(FilmResBlock, self).__init__()
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


class FilmClassifier(nn.Module):
    def __init__(self, filter, last_filter, hidden, last, layer):
        super(FilmClassifier, self).__init__()
        self.conv = nn.Conv2d(filter + 2, last_filter, 1, 1, 0)
        # self.pool = nn.MaxPool2d((input_h, input_w))
        self.mlp = MLP(last_filter, hidden, last, layer)

    def forward(self, x):
        x = positional_encode(x)
        x = self.conv(x).max(2)[0].max(2)[0]
        x = self.mlp(x)
        return x

