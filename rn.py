from torch import nn
import torch
from layers import *
from utils import load_pretrained_embedding


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


def rn_encode(images, questions):
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-1, 1, h).view(1, h, 1, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    images = images.view(n, c, o).transpose(1, 2)
    images = torch.cat([images, x_coordinate, y_coordinate], 2)
    images1 = images.unsqueeze(1).expand(n, o, o, c + 2).contiguous()
    images2 = images.unsqueeze(2).expand(n, o, o, c + 2).contiguous()
    questions = questions.unsqueeze(1).unsqueeze(2).expand(n, o, o, hd)
    # pairs = torch.cat([images1, images2, questions], 3).view(n, o**2, -1)
    pairs = torch.cat([images1, images2, questions], 3)
    return pairs


def lower_sum(relations):
    try:
        device = relations.get_device()
    except:
        device = torch.device('cpu')
    n, h, w, l = relations.size()
    mask = torch.ones([h, w]).tril().view(1, h, w, 1).to(device, dtype=torch.uint8)
    relations = torch.masked_select(relations, mask).view(n, -1, l)
    return relations.sum(1)
