from layers import *
# from utils import *
from utils import load_pretrained_embedding, load_pretrained_conv, rn_encode, lower_sum


class RelationalNetwork(nn.Module):
    def __init__(self, args):
        super(RelationalNetwork, self).__init__()
        # self.filters = args.cv_filter
        # self.layers = args.film_res_layer
        if args.te_pretrained:
            pretrained_weight = load_pretrained_embedding(args.word2idx, args.te_embedding)
        else:
            pretrained_weight = None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer, pretrained_weight)
        if args.cv_pretrained:
            self.visual_encoder = load_pretrained_conv(args.cv_filter)
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.g_theta = MLP((args.cv_filter + 2) * 2 + args.te_hidden, args.rn_gt_hidden, args.rn_gt_hidden, args.rn_gt_layer)
        self.f_phi = MLP(args.rn_gt_hidden, args.rn_fp_hidden, args.a_size, args.rn_fp_layer, args.rn_fp_dropout)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        code = self.text_encoder(question, question_length)
        pairs = rn_encode(x, code)
        relations = self.g_theta(pairs)
        relations = lower_sum(relations)
        logits = self.f_phi(relations)
        return logits
