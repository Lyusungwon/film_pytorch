
def load_default_config(args):
    if args.model == 'film':
        arg_dict = {
            "dataset": "clevr",
            "input_h": 224,
            "input_w": 224,
            "top_k": 0,
            "batch_size": 64,
            "epochs": 200,
            "lr": 3e-4,
            "lr_reduce": True,
            "weight_decay": 1e-5,
            "gradient_clipping": 0,
            "cv_filter": 128,
            "cv_kernel": 3,
            "cv_stride": 2,
            "cv_layer": 4,
            "te_embedding": 200,
            "te_hidden": 4096,
            "te_dropout": 0,
            "te_layer": 1,
            "film_res_kernel": 3,
            "film_res_layer": 4,
            "film_cf_filter": 512,
            "film_fc_hidden": 1024,
            "film_fc_layer": 2,
        }
    elif args.model == 'san':
        arg_dict = {
            "dataset": "vqa2",
            "input_h": 448,
            "input_w": 448,
            "top_k": 1000,
            "batch_size": 100,
            "epochs": 100,
            "lr": 3e-4,
            "lr_reduce": True,
            "weight_decay": 0,
            "gradient_clipping": 0,
            "cv_filter": 512,
            "cv_kernel": 3,
            "cv_stride": 2,
            "cv_layer": 5,
            "te_type": 'lstm',
            "te_embedding": 1024,
            "te_hidden": 1024,
            "te_dropout": 0,
            "te_layer": 1,
            "san_layer": 2,
            "san_k": 512
        }
    elif args.model == 'basern':
        arg_dict = {
            "dataset": "clevr",
            "input_h": 128,
            "input_w": 128,
            "top_k": 0,
            "batch_size": 64,
            "epochs": 100,
            "lr": 2.5e-4,
            "lr_reduce": True,
            "weight_decay": 0,
            "gradient_clipping": 0,
            "cv_filter": 24,
            "cv_kernel": 3,
            "cv_stride": 2,
            "cv_layer": 4,
            "cv_batchnorm": True,
            "te_type": 'lstm',
            "te_embedding": 32,
            "te_hidden": 128,
            "te_dropout": 0,
            "te_layer": 1,
            "basern_gt_hidden": 256,
            "basern_gt_layer": 4,
            "basern_fp_hidden": 256,
            "basern_fp_layer": 3,
            "basern_fp_dropout": 0.5,
        }
    elif args.model == 'rn':
        arg_dict = {
            "dataset": "clevr",
            "input_h": 128,
            "input_w": 128,
            "top_k": 0,
            "batch_size": 64,
            "epochs": 100,
            "lr": 2.5e-4,
            "lr_reduce": True,
            "weight_decay": 0,
            "gradient_clipping": 0,
            "cv_filter": 24,
            "cv_kernel": 3,
            "cv_stride": 2,
            "cv_layer": 4,
            "cv_batchnorm": True,
            "te_type": 'lstm',
            "te_embedding": 32,
            "te_hidden": 128,
            "te_dropout": 0,
            "te_layer": 1,
            "rn_gt_hidden": 256,
            "rn_gt_layer": 4,
            "rn_fp_hidden": 256,
            "rn_fp_layer": 3,
            "rn_fp_dropout": 0.5,
        }
    elif args.model == 'sarn':
        arg_dict = {
            "dataset": "clevr",
            "input_h": 128,
            "input_w": 128,
            "top_k": 0,
            "batch_size": 64,
            "epochs": 100,
            "lr": 2.5e-4,
            "lr_reduce": True,
            "weight_decay": 0,
            "gradient_clipping": 0,
            "cv_filter": 24,
            "cv_kernel": 3,
            "cv_stride": 2,
            "cv_layer": 4,
            "cv_batchnorm": True,
            "te_type": 'lstm',
            "te_embedding": 32,
            "te_hidden": 128,
            "te_dropout": 0,
            "te_layer": 1,
            "sarn_hp_hidden": 256,
            "sarn_hp_layer": 3,
            "sarn_gt_hidden": 256,
            "sarn_gt_layer": 4,
            "sarn_fp_hidden": 256,
            "sarn_fp_layer": 3,
            "sarn_fp_dropout": 0.5,
        }
    elif args.model == 'mrn':
        arg_dict = {
            "dataset": "vqa2",
            "input_h": 224,
            "input_w": 224,
            "top_k": 1000,
            "batch_size": 200,
            "epochs": 100,
            "lr": 2.5e-4,
            "lr_reduce": True,
            "weight_decay": 0,
            "gradient_clipping": 0,
            "te_type": 'gru',
            "te_embedding": 2000,
            "te_hidden": 2400,
            "te_dropout": 0,
            "te_layer": 1,
            "mrn_hidden": 1200,
            "mrn_layer": 3,
        }
    elif args.model == 'mlb':
        arg_dict = {
            "dataset": "vqa2",
            "input_h": 448,
            "input_w": 448,
            "top_k": 2000,
            "batch_size": 64,
            "epochs": 100,
            "lr": 2.5e-4,
            "lr_reduce": True,
            "weight_decay": 0,
            "gradient_clipping": 10.0,
            "cv_filter": 512,
            "cv_kernel": 3,
            "cv_stride": 2,
            "cv_layer": 5,
            "te_type": 'gru',
            "te_embedding": 200,
            "te_hidden": 2400,
            "te_dropout": 0.5,
            "te_layer": 1,
            "mlb_hidden": 1200,
            "mlb_glimpse": 2,
        }
    ori_dict = vars(args)
    for key in arg_dict.keys():
        if ori_dict[key] is not None:
            arg_dict[key] = ori_dict[key]
    ori_dict.update(arg_dict)
    print(f"Default config for {args.model} loaded.")
    return args
