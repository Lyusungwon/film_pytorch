import os
import torch
import argparse
import datetime
from configloader import load_default_config
from pathlib import Path
from film import Film
from san import San
from rn import RelationalNetwork
from mrn import Mrn
from mlb import Mlb
from basern import BaseRN
from sarn import Sarn

home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--project', type=str, default='vqa')
    parser.add_argument('--model', type=str, choices=['basern', 'rn', 'sarn', 'san', 'mrn', 'mlb', 'film'])

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-directory', type=str, default=os.path.join(home, 'data'), help='directory of data')
    data_arg.add_argument('--dataset', type=str)
    data_arg.add_argument('--input-h', type=int)
    data_arg.add_argument('--input-w', type=int)
    data_arg.add_argument('--top-k', type=int)
    data_arg.add_argument('--multi-label', action='store_true')

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--batch-size', type=int)
    train_arg.add_argument('--epochs', type=int)
    train_arg.add_argument('--lr', type=float)
    train_arg.add_argument('--lr-reduce', action='store_true')
    train_arg.add_argument('--weight-decay', type=float)
    train_arg.add_argument('--gradient-clipping', type=float)
    train_arg.add_argument('--log-directory', type=str, default=os.path.join(home, 'experiment'), metavar='N', help='log directory')
    train_arg.add_argument('--device', type=int, default=0, metavar='N', help='gpu number')
    train_arg.add_argument('--cpu-num', type=int, default=8, metavar='N', help='number of cpu')
    train_arg.add_argument('--multi-gpu', action='store_true')
    train_arg.add_argument('--gpu-num', type=int, default=4, metavar='N', help='number of cpu')
    train_arg.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    train_arg.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    train_arg.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N', help='time of the run(no modify)')
    train_arg.add_argument('--memo', type=str, default='default', metavar='N', help='memo of the model')
    train_arg.add_argument('--load-model', type=str, default=None, help='load previous model')

    model_arg = parser.add_argument_group('Model')
    # Convolution
    model_arg.add_argument('--cv-pretrained', action='store_true')
    model_arg.add_argument('--cv-filter', type=int)
    model_arg.add_argument('--cv-kernel', type=int)
    model_arg.add_argument('--cv-stride', type=int)
    model_arg.add_argument('--cv-layer', type=int)
    model_arg.add_argument('--cv-batchnorm', action='store_true')
    # Text Encoder
    model_arg.add_argument('--te-pretrained', action='store_true')
    model_arg.add_argument('--te-type', type=str, choices=['gru', 'lstm'])
    model_arg.add_argument('--te-embedding', type=int)
    model_arg.add_argument('--te-hidden', type=int)
    model_arg.add_argument('--te-layer', type=int)
    model_arg.add_argument('--te-dropout', type=float)
    # film
    model_arg.add_argument('--film-res-kernel', type=int)
    model_arg.add_argument('--film-res-layer', type=int)
    model_arg.add_argument('--film-cf-filter', type=int)
    model_arg.add_argument('--film-fc-hidden', type=int)
    model_arg.add_argument('--film-fc-layer', type=int)
    # san
    model_arg.add_argument('--san-layer', type=int)
    model_arg.add_argument('--san-k', type=int)
    # basern
    model_arg.add_argument('--basern-gt-hidden', type=int)
    model_arg.add_argument('--basern-gt-layer', type=int)
    model_arg.add_argument('--basern-fp-hidden', type=int)
    model_arg.add_argument('--basern-fp-layer', type=int)
    model_arg.add_argument('--basern-fp-dropout', type=float)
    # rn
    model_arg.add_argument('--rn-gt-hidden', type=int)
    model_arg.add_argument('--rn-gt-layer', type=int)
    model_arg.add_argument('--rn-fp-hidden', type=int)
    model_arg.add_argument('--rn-fp-layer', type=int)
    model_arg.add_argument('--rn-fp-dropout', type=float)
    # rn
    model_arg.add_argument('--sarn-hp-hidden', type=int)
    model_arg.add_argument('--sarn-hp-layer', type=int)
    model_arg.add_argument('--sarn-gt-hidden', type=int)
    model_arg.add_argument('--sarn-gt-layer', type=int)
    model_arg.add_argument('--sarn-fp-hidden', type=int)
    model_arg.add_argument('--sarn-fp-layer', type=int)
    model_arg.add_argument('--sarn-fp-dropout', type=float)
    # mrn
    model_arg.add_argument('--mrn-hidden', type=int)
    model_arg.add_argument('--mrn-layer', type=int)
    # mln
    model_arg.add_argument('--mlb-hidden', type=int)
    model_arg.add_argument('--mlb-glimpse', type=int)

    args, unparsed = parser.parse_known_args()
    args = load_default_config(args)

    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.device)
        args.device = torch.device(args.device)

    args.data_config = [args.input_h, args.input_w, args.cpu_num, args.cv_pretrained, args.top_k, args.multi_label]

    config_list = [args.project, args.model, args.dataset, args.epochs, args.batch_size, args.lr,
                   args.weight_decay, args.gradient_clipping,
                   args.device, args.multi_gpu, args.gpu_num] + args.data_config + \
        ['cv', args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm,
         'te', args.te_pretrained, args.te_type, args.te_embedding, args.te_hidden, args.te_layer, args.te_dropout]

    if args.model == 'film':
        config_list = config_list + \
            ['film', args.film_res_kernel, args.film_res_layer,
             args.film_cf_filter, args.film_fc_hidden, args.film_fc_layer,
             args.memo]
        model = Film(args)
    elif args.model == 'san':
        config_list = config_list + \
            ['san', args.san_layer, args.san_k,
             args.memo]
        model = San(args)
    elif args.model == 'basern':
        config_list = config_list + \
            ['basern', args.basern_gt_hidden, args.basern_gt_layer, args.basern_fp_hidden, args.basern_fp_layer, args.basern_fp_dropout,
             args.memo]
        model = BaseRN(args)
    elif args.model == 'rn':
        config_list = config_list + \
            ['rn', args.rn_gt_hidden, args.rn_gt_layer, args.rn_fp_hidden, args.rn_fp_layer, args.rn_fp_dropout,
             args.memo]
        model = RelationalNetwork(args)
    elif args.model == 'sarn':
        config_list = config_list + \
            ['sarn', args.sarn_hp_hidden, args.sarn_hp_layer, args.sarn_gt_hidden, args.sarn_gt_layer, args.sarn_fp_hidden, args.sarn_fp_layer, args.sarn_fp_dropout,
             args.memo]
        model = Sarn(args)
    elif args.model == 'mrn':
        config_list = config_list + \
            ['mrn', args.mrn_hidden, args.mrn_layer, args.memo]
        model = Mrn(args)
    elif args.model == 'mlb':
        config_list = config_list + \
            ['mlb', args.mlb_hidden, args.mlb_glimpse, args.memo]
        model = Mlb(args)
    else:
        print("Not an available model.")

    args.config = '_'.join(map(str, config_list))
    if args.load_model:
        args.log = os.path.join(args.log_directory, args.project, args.load_model)
        args.timestamp = args.load_model[:12]
    else:
        args.log = os.path.join(args.log_directory, args.project, args.timestamp + args.config)

    print(f"Config: {args.config}")

    return args, model
