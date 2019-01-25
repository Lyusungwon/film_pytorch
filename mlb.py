import torch
from layers import *
from utils import *


class Mlb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if not args.cv_pretrained:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        q = args.te_hidden
        i = args.cv_filter
        h = args.mlb_hidden
        g = args.mlb_glimpse
        o = args.a_size
        self.Uq = nn.Sequential(
            nn.Linear(q, h),
            nn.Tanh()
        )
        self.Vf = nn.Sequential(
            nn.Linear(i, h),
            nn.Tanh(),
        )
        self.P1 = nn.Sequential(
            nn.Linear(h, g),
            nn.Softmax(dim=1)
        )
        self.Wq = nn.Sequential(
            nn.Linear(q, h),
            nn.Tanh()
        )
        self.Vv = nn.Sequential(
            nn.Linear(g * h, h),
            nn.Tanh(),
        )
        self.P2 = nn.Linear(h, o)

    def forward(self, image, question, question_length):
        if self.cv_pretrained:
            i = image
        else:
            i = self.visual_encoder(image)
        b, c, h, w = i.size()
        i = i.view(b, c, -1).transpose(1, 2) # b o c
        _, q = self.text_encoder(question, question_length)
        i1 = self.Vf(i) # b o h
        q1 = self.Uq(q).unsqueeze(1) # b 1 h
        f = self.P1(i1 * q1).transpose(1, 2) # b g o
        i2 = torch.matmul(f, i1).view(b, -1) # b g*h
        i3 = self.Vv(i2) # b h
        q2 = self.Wq(q) # b h
        logits = self.P2(i3 * q2).squeeze(1) # b o
        return logits
