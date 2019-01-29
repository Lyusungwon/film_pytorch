import torch
from layers import *
from utils import *
import numpy as np
from utils import load_pretrained_embedding


class Mrn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_type, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            raise
        else:
            self.visual_encoder = load_pretrained_conv()
            filters = 2048 if args.dataset == 'vqa2' else 1024
        self.first_block = MrnBlock(filters, args.te_hidden, args.mrn_hidden)
        self.blocks = nn.ModuleList([MrnBlock(filters, args.mrn_hidden, args.mrn_hidden) for _ in range(args.mrn_layer - 1)])
        self.fc = nn.Linear(args.mrn_hidden, args.a_size)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image).squeeze(3).squeeze(2)
        _, q = self.text_encoder(question, question_length)
        h = self.first_block(q, x)
        for block in self.blocks:
            h = block(h, x)
        logits = self.fc(h)
        return logits


class MrnBlock(nn.Module):
    def __init__(self, i, q, h):
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
        self.res = nn.Linear(q, h)

    def forward(self, q, i):
        question = self.qs(q)
        objects = self.vb(i)
        h = objects * question + self.res(q)
        return h
