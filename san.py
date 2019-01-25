import torch
from layers import *
from utils import *


class San(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            filters = 1024 if args.dataset == 'clevr' else 2048
            self.visual_encoder = nn.Conv2d(filters, args.cv_filter, 1, 1)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.blocks = nn.ModuleList([SanBlock(args.cv_filter, args.te_hidden, args.san_k) for _ in range(args.san_layer)])
        self.fc = nn.Linear(args.cv_filter, args.a_size)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        _, u = self.text_encoder(question, question_length)
        for block in self.blocks:
            u = block(x, u)
        logits = self.fc(u)
        return logits


class SanBlock(nn.Module):
    def __init__(self, id, qd, k):
        super().__init__()
        self.wia = nn.Linear(id, k, bias=False)
        self.wqa = nn.Linear(qd, k)
        self.wp = nn.Linear(k, 1)

    def forward(self, i, q):
        wi = self.wia(i.transpose(2, 1))
        wq = self.wqa(q).unsqueeze(1)
        ha = torch.tanh(wi + wq)
        pi = torch.softmax(self.wp(ha), dim=1)
        u = torch.matmul(i, pi).squeeze(2) + q
        return u
