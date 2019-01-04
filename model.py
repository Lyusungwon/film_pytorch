from layers import *


class Film(nn.Module):
    def __init__(self, args):
        super(Film, self).__init__()
        self.filters = args.cv_filter
        self.layers = args.res_layer
        self.text_encoder = Text_encoder(args.q_size, args.te_embedding, args.te_hidden, args.te_layer)
        self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
        self.fc = nn.Linear(args.te_hidden, args.cv_filter * args.res_layer * 2)
        self.res_blocks = nn.ModuleList([ResBlock(args.cv_filter, args.cv_kernel) for _ in range(args.res_layer)])
        self.classifier = Classifier(args.cv_filter, args.cf_filter, args.fc_hidden, args.a_size, args.fc_layer)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        code = self.text_encoder(question, question_length)
        betagamma = self.fc(code).view(-1, self.layers, 2, self.filters)
        for n, block in enumerate(self.res_blocks):
            x = block(x, betagamma[:, n])
        logits = self.classifier(x)
        return logits

