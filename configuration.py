import os
import torch
import argparse
import datetime
from configloader import load_default_config
from pathlib import Path
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser(description='parser')

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--project', type=str, default='film')
    model_arg.add_argument('--model', type=str, default='film')
    # Load configuration of the paper
    model_arg.add_argument('--load-default', action='store_true')
    # Convolution
    model_arg.add_argument('--cv-pretrained', action='store_true')
    model_arg.add_argument('--cv-filter', type=int, default=512)
    model_arg.add_argument('--cv-kernel', type=int, default=4)
    model_arg.add_argument('--cv-stride', type=int, default=2)
    model_arg.add_argument('--cv-layer', type=int, default=4)
    model_arg.add_argument('--cv-batchnorm', action='store_false')
    # Text Encoder
    model_arg.add_argument('--te-pretrained', action='store_true')
    model_arg.add_argument('--te-embedding', type=int, default=200)
    model_arg.add_argument('--te-hidden', type=int, default=512)
    model_arg.add_argument('--te-layer', type=int, default=1)
    # film
    model_arg.add_argument('--film-res-kernel', type=int, default=3)
    model_arg.add_argument('--film-res-layer', type=int, default=4)
    model_arg.add_argument('--film-cf-filter', type=int, default=512)
    model_arg.add_argument('--film-fc-hidden', type=int, default=1024)
    model_arg.add_argument('--film-fc-layer', type=int, default=2)
    # san
    model_arg.add_argument('--san-layer', type=int, default=2)
    model_arg.add_argument('--san-k', type=int, default=640)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-directory', type=str, default=os.path.join(home,'data'), metavar='N', help='directory of data')
    data_arg.add_argument('--dataset', type=str, default='clevr')
    data_arg.add_argument('--input-h', type=int, default=128)
    data_arg.add_argument('--input-w', type=int, default=128)

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    train_arg.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    train_arg.add_argument('--lr', type=float, default=3e-4, metavar='N', help='learning rate (default: 3e-4)')
    train_arg.add_argument('--weight-decay', type=float, default=1e-5)
    train_arg.add_argument('--log-directory', type=str, default=os.path.join(home, 'experiment'), metavar='N', help='log directory')
    train_arg.add_argument('--device', type=int, default=0, metavar='N', help='gpu number')
    train_arg.add_argument('--cpu-num', type=int, default=0, metavar='N', help='number of cpu')
    model_arg.add_argument('--multi-gpu', action='store_true')
    train_arg.add_argument('--gpu-num', type=int, default=4, metavar='N', help='number of cpu')
    train_arg.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    train_arg.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    train_arg.add_argument('--time-stamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N', help='time of the run(no modify)')
    train_arg.add_argument('--memo', type=str, default='default', metavar='N', help='memo of the model')
    train_arg.add_argument('--load-model', type=str, default=None, metavar='N', help='load previous model')

    args, unparsed = parser.parse_known_args()

    if args.load_default:
        args = load_default_config(args)

    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.device)
        args.device = torch.device(args.device)

    args.data_config = [args.input_h, args.input_w, args.cpu_num]

    config_list = [args.project, args.model, args.dataset, args.epochs, args.batch_size, args.lr,
                   args.device, args.multi_gpu, args.gpu_num] + args.data_config + \
                ['cv', args.cv_pretrained, args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm,
                 'te', args.te_pretrained, args.te_embedding, args.te_hidden, args.te_layer]

    if args.model == 'film':
        config_list = config_list + \
            ['film', args.film_res_kernel, args.film_res_layer,
             args.film_cf_filter, args.film_fc_hidden, args.film_fc_layer,
             args.memo]
    elif args.model == 'san':
        config_list = config_list + \
            ['san', args.san_layer, args.san_k,
             args.memo]

    args.config = '_'.join(map(str, config_list))
    if args.load_model:
        args.log = os.path.join(args.log_directory, args.project, args.load_model)
        args.time_stamp = args.load_model[:12]
    else:
        args.log = os.path.join(args.log_directory, args.project, args.time_stamp + args.config)

    print("Config:", args.config)

    return args
