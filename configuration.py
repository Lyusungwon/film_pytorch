import torch
import argparse
import datetime
from pathlib import Path
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser(description='parser')

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--project', type=str, default='rn')
    model_arg.add_argument('--model', type=str, default='baseline')
    # Convolution
    model_arg.add_argument('--cv-filter', type=int, default=32)
    model_arg.add_argument('--cv-kernel', type=int, default=3)
    model_arg.add_argument('--cv-stride', type=int, default=2)
    model_arg.add_argument('--cv-layer', type=int, default=4)
    model_arg.add_argument('--cv-layernorm', action='store_false')
    # Text Encoder
    model_arg.add_argument('--te-embedding', type=int, default=1)
    model_arg.add_argument('--te-hidden', type=int, default=128)
    model_arg.add_argument('--te-layer', type=int, default=1)
    # h psi
    model_arg.add_argument('--hp-hidden', type=int, default=128)
    model_arg.add_argument('--hp-layer', type=int, default=3)
    # g theta
    model_arg.add_argument('--gt-hidden', type=int, default=128)
    model_arg.add_argument('--gt-layer', type=int, default=3)
    # f phi
    model_arg.add_argument('--fp-hidden', type=int, default=128)
    model_arg.add_argument('--fp-dropout', type=int, default=5)
    model_arg.add_argument('--fp-dropout-rate', type=float, default=0.2)
    model_arg.add_argument('--fp-layer', type=int, default=3)
    # att
    model_arg.add_argument('--attn-head', type=int, default=1)
    model_arg.add_argument('--attn-key', type=int, default=32)
    model_arg.add_argument('--attn-val', type=int, default=32)
    # film
    model_arg.add_argument('--film-lstm-hidden', type=int, default=16)
    model_arg.add_argument('--film-kernel', type=int, default=3)
    model_arg.add_argument('--film-res-layer', type=int, default=5)
    model_arg.add_argument('--film-last-filter', type=int, default=64)
    model_arg.add_argument('--film-mlp-hidden', type=int, default=128)
    model_arg.add_argument('--film-mlp-layer', type=int, default=2)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-directory', type=str, default = home + '/data/', metavar='N', help='directory of data')
    data_arg.add_argument('--dataset', type=str, default='sortofclevr')
    data_arg.add_argument('--train-size', type=int, default=9800)
    data_arg.add_argument('--test-size', type=int, default=200)
    data_arg.add_argument('--image-size', type=int, default=75)
    data_arg.add_argument('--size', type=int, default=5)
    data_arg.add_argument('--closest', type=int, default=3)
    data_arg.add_argument('--channel-size', type=int, default=3)
    data_arg.add_argument('--input-h', type=int, default=75)
    data_arg.add_argument('--input-w', type=int, default=75)

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    train_arg.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
    train_arg.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate (default: 1e-4)')
    train_arg.add_argument('--log-directory', type=str, default = home + '/experiment/', metavar='N', help='log directory')
    train_arg.add_argument('--device', type=int, default=0, metavar='N', help='number of cuda')
    train_arg.add_argument('--cpu-num', type=int, default=0, metavar='N', help='number of cpu')
    train_arg.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    train_arg.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    train_arg.add_argument('--time-stamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N', help='time of the run(no modify)')
    train_arg.add_argument('--memo', type=str, default='default', metavar='N', help='memo of the model')
    train_arg.add_argument('--load-model', type=str, default='000000000000', metavar='N', help='load previous model')
    train_arg.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start-epoch number')

    args, unparsed = parser.parse_known_args()

    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.device)
        args.device = torch.device(args.device)

    if args.dataset == 'clevr':
        args.data_config = [args.input_h, args.input_w, args.cpu_num]
    else:
        args.data_config = [args.train_size, args.test_size, args.image_size, args.size, args.closest]
        args.input_h = args.image_size
        args.input_w = args.image_size

    config_list = [args.project, args.model, args.dataset, args.epochs, args.batch_size, args.lr, args.device,
                   'inp', args.channel_size] + args.data_config + \
                  ['cv', args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_layernorm,
                   'te', args.te_embedding, args.te_hidden, args.te_layer,
                   'hp', args.hp_hidden, args.hp_layer,
                   'gt', args.gt_hidden, args.gt_layer,
                   'fp', args.fp_hidden, args.fp_dropout, args.fp_dropout_rate, args.fp_layer,
                   args.memo]

    args.config = '_'.join(map(str, config_list))
    args.log = args.log_directory + args.project + '/' + args.time_stamp + args.config + '/'
    print("Config:", args.config)

    return args
