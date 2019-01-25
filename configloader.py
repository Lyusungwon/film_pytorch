
def load_default_config(args):
    if args.model == 'film' and args.dataset == 'clevr':
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
        args.top_k = 0
        args.lr_reduce = True
    elif args.model == 'san' and args.dataset == 'vqa2':
        args.cv_pretrained = True
        args.cv_filter = 1024
        args.te_embedding = 1024
        args.te_hidden = 1024
        args.san_layer = 2
        args.san_k = 1024
        args.batch_size = 100
        args.input_h = 448
        args.input_w = 448
        args.top_k = 1000
        args.lr_reduce = True
    elif args.model == 'rn' and args.dataset == 'clevr':
        args.cv_filter = 24
        args.cv_kernel = 3
        args.cv_stride = 2
        args.cv_layer = 4
        args.cv_batchnorm = True
        args.te_embedding = 32
        args.te_hidden = 128
        args.te_layer = 1
        args.rn_gt_hidden = 256
        args.rn_gt_layer = 4
        args.rn_fp_hidden = 256
        args.rn_fp_layer = 3
        args.rn_fp_dropout = 0.5
        args.input_h = 128
        args.input_w = 128
        args.batch_size = 64
        args.lr = 2.5e-4
        args.lr_reduce = True
        args.top_k = 0
    elif args.model == 'mrn' and args.dataset == 'vqa2':
        args.cv_pretrained = True
        args.cv_filter = 2048
        args.te_embedding = 200
        args.te_hidden = 2400
        args.mrn_hidden = 1200
        args.mrn_layer = 3
        args.batch_size = 200
        args.input_h = 448
        args.input_w = 448
        args.top_k = 1000
        args.lr_reduce = True
    elif args.model == 'mlb' and args.dataset == 'vqa2':
        args.cv_pretrained = True
        args.cv_filter = 2048
        args.te_embedding = 200
        args.te_hidden = 2400
        args.te_dropout = 0.5
        args.mlb_hidden = 1200
        args.mlb_glimpse = 2
        args.batch_size = 100
        args.lr = 3e-4
        args.input_h = 448
        args.input_w = 448
        args.top_k = 2000
        args.lr_reduce = True
        args.gradient_clipping = 10.0
    else:
        print("Default config not found!")
        return args
    print(f"Default config for {args.model} loaded.")
    return args
