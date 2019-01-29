from torch import nn
from layers import *
from utils import load_pretrained_embedding, rn_encode, lower_sum


class RelationalNetwork(nn.Module):
    def __init__(self, args):
        super(RelationalNetwork, self).__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            filters = 2048 if args.dataset == 'vqa2' else 1024
            self.visual_resize = nn.Conv2d(filters, args.cv_filter, 3, 1, 1)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.g_theta = MLP((args.cv_filter + 2) * 2 + args.te_hidden, args.rn_gt_hidden, args.rn_gt_hidden, args.rn_gt_layer)
        self.f_phi = MLP(args.rn_gt_hidden, args.rn_fp_hidden, args.a_size, args.rn_fp_layer, args.rn_fp_dropout)

    def forward(self, image, question, question_length):
        if not self.cv_pretrained:
            image = image * 2 - 1
        x = self.visual_encoder(image)
        _, code = self.text_encoder(question, question_length)
        pairs = rn_encode(x, code)
        relations = self.g_theta(pairs)
        relations = lower_sum(relations)
        logits = self.f_phi(relations)
        return logits
