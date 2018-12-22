import model
import numpy as np
import torch

def build_model(args):
    device = args.device
    torch.manual_seed(args.seed)
    models = dict()
    cv_layout = [(args.cv_filter, args.cv_kernel, args.cv_stride) for i in range(args.cv_layer)]
    if args.model == 'baseline':
        gt_layout = [(args.cv_filter + 2) + args.te_embedding * 2] + [args.gt_hidden for i in range(args.gt_layer)]
        fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 1)] + [
            args.label_size]
        conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
        g_theta = model.MLP(gt_layout).to(device)
        f_phi = model.MLP(fp_layout).to(device)
        models['g_theta.pt'] = g_theta
        models['f_phi.pt'] = f_phi

    elif args.model == 'rn':
        gt_layout = [(args.cv_filter + 2) * 2 + args.te_embedding * 2] + [args.gt_hidden for i in range(args.gt_layer)]
        fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 1)] + [
            args.label_size]
        conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
        g_theta = model.MLP(gt_layout).to(device)
        f_phi = model.MLP(fp_layout).to(device)
        models['g_theta.pt'] = g_theta
        models['f_phi.pt'] = f_phi

    elif args.model == 'sarn':
        gt_layout = [2 * (args.cv_filter + 2 + args.te_embedding)] + [args.gt_hidden for i in range(args.gt_layer)]
        hp_layout = [args.cv_filter + 2 + args.te_embedding * 2] + [args.hp_hidden for i in
                                                                    range(args.hp_layer - 1)] + [1]
        fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 1)] + [
            args.label_size]
        conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
        g_theta = model.MLP(gt_layout).to(device)
        h_psi = model.MLP(hp_layout).to(device)
        f_phi = model.MLP(fp_layout).to(device)
        models['g_theta.pt'] = g_theta
        models['f_phi.pt'] = f_phi
        models['h_psi.pt'] = h_psi

    elif args.model == 'sarn_att':
        gt_layout = [args.cv_filter + 2] + [args.gt_hidden for i in range(args.gt_layer)]
        hp_layout = [args.cv_filter + 2 + args.te_embedding * 2] + [args.hp_hidden for i in
                                                                    range(args.hp_layer - 1)] + [1]
        fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 2)] + [
            args.label_size]
        conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
        g_theta = model.MLP(gt_layout).to(device)
        h_psi = model.MLP(hp_layout).to(device)
        attn = model.MultiHeadAttention(n_head=args.attn_head, d_model=args.cv_filter + 2, d_k=args.attn_key, d_v=args.attn_val).to(device)
        f_phi = model.MLP(fp_layout).to(device)
        models['g_theta.pt'] = g_theta
        models['f_phi.pt'] = f_phi
        models['h_psi.pt'] = h_psi
        models['attn.pt'] = attn

    elif args.model == 'film':
        conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
        input_h, input_w = args.input_h, args.input_w
        for i in range(args.cv_layer):
            input_h = int(np.ceil(input_h / 2))
            input_w = int(np.ceil(input_w / 2))
        film = model.Film(args.te_embedding * 2, args.film_lstm_hidden, args.cv_filter, args.film_kernel,
                          args.film_res_layer, args.film_last_filter, input_h, input_w, args.film_mlp_hidden,
                          args.film_mlp_layer, args.label_size).to(device)
        models['film.pt'] = film

    if args.dataset == 'clevr':
        text_encoder = model.Text_encoder(args.q_size, args.te_embedding, args.te_hidden,
                                          args.te_layer).to(device)
    else:
        text_encoder = model.Text_embedding(args.c_size, args.q_size,
                                            args.te_embedding).to(device)
    models['text_encoder.pt'] = text_encoder
    models['conv.pt'] = conv
    return models
