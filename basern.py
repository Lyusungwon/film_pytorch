from torch import nn
import torch
from layers import *
from utils import load_pretrained_embedding


class BaseRN(nn.Module):
    def __init__(self, args):
        super(BaseRN, self).__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_type, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            filters = 2048 if args.dataset == 'vqa2' else 1024
            self.visual_resize = nn.Conv2d(filters, args.cv_filter, 3, 1, 1)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.g_theta = MLP(args.cv_filter + 2 + args.te_hidden, args.basern_gt_hidden, args.basern_gt_hidden, args.basern_gt_layer)
        self.f_phi = MLP(args.basern_gt_hidden, args.basern_fp_hidden, args.a_size, args.basern_fp_layer, args.basern_fp_dropout)

    def forward(self, image, question, question_length):
        if not self.cv_pretrained:
            image = image * 2 - 1
        x = self.visual_encoder(image)
        _, code = self.text_encoder(question, question_length)
        pairs = baseline_encode(x, code)
        relations = self.g_theta(pairs).sum(1)
        logits = self.f_phi(relations)
        return logits


def baseline_encode(images, questions):
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    questions = questions.unsqueeze(2).unsqueeze(3).expand(n, hd, h, w)
    images = torch.cat([images, x_coordinate, y_coordinate, questions], 1).view(n, -1, o).transpose(1, 2)
    return images
