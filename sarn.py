from torch import nn
import torch
from layers import *
from utils import load_pretrained_embedding
import torch.nn.functional as F

class Sarn(nn.Module):
    def __init__(self, args):
        super(Sarn, self).__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_type, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            filters = 2048 if args.dataset == 'vqa2' else 1024
            self.visual_resize = nn.Conv2d(filters, args.cv_filter, 3, 1, 1)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.h_psi = MLP(args.cv_filter + 2 + args.te_hidden, args.sarn_hp_hidden, 1, args.sarn_hp_layer)
        self.g_theta = MLP((args.cv_filter + 2) * 2 + args.te_hidden, args.sarn_gt_hidden, args.sarn_gt_hidden, args.sarn_gt_layer)
        self.f_phi = MLP(args.sarn_gt_hidden, args.sarn_fp_hidden, args.a_size, args.sarn_fp_layer, args.sarn_fp_dropout)

    def forward(self, image, question, question_length):
        if not self.cv_pretrained:
            image = image * 2 - 1
            x = self.visual_encoder(image)
        else:
            x = self.visual_resize(image)
        _, code = self.text_encoder(question, question_length)
        coordinate_encoded, question_encoded = sarn_encode(x, code)
        attention = self.h_psi(question_encoded)
        pairs = sarn_pair(coordinate_encoded, question_encoded, attention)
        relations = self.g_theta(pairs).sum(1)
        logits= self.f_phi(relations)
        return logits


def sarn_encode(objects, code):
    try:
        device = relations.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = objects.size()
    o = h * w
    hd = code.size(1)
    x_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    coordinate_encoded = torch.cat([objects, x_coordinate, y_coordinate], 1)
    question = code.view(n, hd, 1, 1).expand(n, hd, h, w)
    question_encoded = torch.cat([coordinate_encoded, question], 1).view(n, -1, o).transpose(1, 2)
    return coordinate_encoded.view(n, -1, o).transpose(1, 2), question_encoded


def sarn_pair(coordinate_encoded, question_encoded, attention):
    selection = F.softmax(attention.squeeze(2), dim=1)
    selected = torch.bmm(selection.unsqueeze(1), coordinate_encoded).expand_as(coordinate_encoded)
    pairs = torch.cat([question_encoded, selected], 2)
    return pairs
