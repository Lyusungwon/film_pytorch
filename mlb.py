import torch
from layers import *
from utils import *
# from utils import load_pretrained_embedding, load_pretrained_conv


class MLB(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.te_pretrained:
            pretrained_weight = load_pretrained_embedding(args.word2idx, args.te_embedding)
        else:
            pretrained_weight = None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, pretrained_weight)
        if args.cv_pretrained:
            self.visual_encoder = load_pretrained_conv(args.cv_filter)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.blocks = nn.ModuleList([MLBBlock(args.cv_filter, args.te_hidden, args.mlb_hidden) for _ in range(args.mlb_layer)])
        self.fc = nn.Linear(args.mlb_hidden, args.a_size)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        x = x.transpose(0, 3, 1, 2)
        b, h, w, c = x.size()
        objects = x.view(b, -1, c)
        h = self.text_encoder(question, question_length)
        for block in self.blocks:
            h = block(h, objects)
        logits = self.fc(u)
        return logits


class MLBBlock(nn.Module):
    def __init__(self, i, q, h):
        super().__init__()
        self.vb = VisualBlock(i, h)
        self.qs = nn.Sequential(
            nn.Linear(q, h),
            nn.Tanh(inplace=True)
        )
        self.res = nn.Linear(q, h)

    def forward(self, q, i):
        res = self.res(q)
        question = self.qs(q)
        objects = self.vb(i)
        h = res + objects * question
        return h


class VisualBlock(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.net = nn.Squential(
            nn.Linear(i, h),
            nn.Tanh(inplace=True),
            nn.Linear(h, h),
            nn.Tanh(inplace=True)
        )

    def forward(self, i):
        return self.net(i)

