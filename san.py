import torch
from layers import *
from utils import *
# from utils import load_pretrained_embedding, load_pretrained_conv


class San(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.cv_filter == args.te_hidden
        self.d = args.cv_filter
        if args.te_pretrained:
            pretrained_weight = load_pretrained_embedding(args.word2idx, args.te_embedding)
        else:
            pretrained_weight = None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, pretrained_weight)
        if args.cv_pretrained:
            self.visual_encoder = load_pretrained_conv(args.cv_filter)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.blocks = nn.ModuleList([SanBlock(self.d, args.san_k) for _ in range(args.san_layer)])
        self.fc = nn.Linear(self.d, args.a_size)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        u = self.text_encoder(question, question_length)
        for block in self.blocks:
            _, u = block(x, u)
        logits = self.fc(u)
        return logits


class SanBlock(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.wqa = nn.Linear(d, k)
        self.wia = nn.Linear(d, k, bias=False)
        self.wp = nn.Linear(k, 1)

    def forward(self, i, q):
        wi = self.wia(i.transpose(1, 2))
        wq = self.wqa(q).unsqueeze(1).expand_as(wi)
        ha = torch.tanh(wi + wq)
        pi = torch.softmax(self.wp(ha), dim=1)
        u = torch.matmul(i, pi).squeeze(2).squeeze(1) + q
        return i, u

