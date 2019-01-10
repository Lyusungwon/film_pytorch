
def load_default_config(args):
    if args.model == 'film':
        args.cv_pretrained = True
        args.cv_filter = 128
        args.cv_kernel = 4
        args.cv_stride = 2
        args.cv_layer = 4
        args.te_embedding = 200
        args.te_hidden = 4096
        args.te_layer = 1
        args.film_res_kernel = 3
        args.film_res_layer = 4
        args.film_cf_filter = 512
        args.film_fc_hidden = 1024
        args.film_fc_layer = 2
        args.input_h = 224
        args.input_w = 224
        args.batch_size = 64
        args.lr = 3e-4
        args.weight_decay = 1e-5
        return args
    elif args.model == 'san':
        args.cv_pretrained = True
        args.cv_filter = 512
        args.te_hidden = 512
        args.san_layer = 1
        args.san_k = 640
        args.batch_size = 100
        return args

