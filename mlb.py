import torch
from layers import *
from utils import *
import numpy as np
# from utils import load_pretrained_embedding


class Mlb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word2idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, pretrained_weight)
        if not args.cv_pretrained:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
            input_h, input_w = args.input_h, args.input_w
            for _ in range(args.cv_layer):
                input_h = int(np.ceil(input_h / args.cv_stride))
                input_w = int(np.ceil(input_w / args.cv_stride))
            object_num = input_h * input_w
        else:
            object_num = 64
        self.blocks = nn.ModuleList([MlbBlock(args.cv_filter, args.te_hidden, args.mlb_hidden, object_num) for _ in range(args.mlb_layer)])
        self.fc = nn.Linear(args.mlb_hidden, args.a_size)

    def forward(self, image, question, question_length):
        if self.cv_pretrained:
            x = image
        else:
            x = self.visual_encoder(image)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)
        h = self.text_encoder(question, question_length)
        for block in self.blocks:
            h = block(h, x)
        logits = self.fc(u)
        return logits


class MlbBlock(nn.Module):
    def __init__(self, i, q, h, object_number):
        super().__init__()
        self.qs = nn.Sequential(
            nn.Linear(q, h),
            nn.Tanh()
        )
        self.vb = nn.Sequential(
            nn.Linear(i, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh()
        )
        self.fc = nn.Linear(object_num, 1)
        self.res = nn.Linear(q, h)

    def forward(self, q, i):
        objects = self.vb(i).transpose(1, 2)
        question = self.qs(q).unsqueeze(2)
        f = objects * question
        res = self.res(q)
        h = res + self.fc(f)
        return h
