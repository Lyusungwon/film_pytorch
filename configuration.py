import os
import torch
import argparse
import datetime
from pathlib import Path
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser(description='parser')

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--project', type=str, default='film')
    model_arg.add_argument('--model', type=str, default='film')
    # Convolution
    model_arg.add_argument('--cv-filter', type=int, default=128)
    model_arg.add_argument('--cv-kernel', type=int, default=3)
    model_arg.add_argument('--cv-stride', type=int, default=2)
    model_arg.add_argument('--cv-layer', type=int, default=4)
    model_arg.add_argument('--cv-batchnorm', action='store_false')
    # Text Encoder
    model_arg.add_argument('--te-embedding', type=int, default=200)
    model_arg.add_argument('--te-hidden', type=int, default=4096)
    model_arg.add_argument('--te-layer', type=int, default=1)
    # film
    model_arg.add_argument('--res-layer', type=int, default=4)
    # classifier
    model_arg.add_argument('--cf-filter', type=int, default=512)
    model_arg.add_argument('--fc-hidden', type=int, default=1024)
    model_arg.add_argument('--fc-layer', type=int, default=2)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-directory', type=str, default=os.path.join(home,'data'), metavar='N', help='directory of data')
    data_arg.add_argument('--dataset', type=str, default='clevr')
    data_arg.add_argument('--input-h', type=int, default=224)
    data_arg.add_argument('--input-w', type=int, default=224)

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    train_arg.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
    train_arg.add_argument('--lr', type=float, default=3e-4, metavar='N', help='learning rate (default: 1e-4)')
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
    train_arg.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start-epoch number')

    args, unparsed = parser.parse_known_args()

    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.device)
        args.device = torch.device(args.device)

    args.data_config = [args.input_h, args.input_w, args.cpu_num]

    config_list = [args.project, args.model, args.dataset, args.epochs, args.batch_size, args.lr, args.device, args.multi_gpu, args.gpu_num] + \
                  args.data_config + \
                  ['cv', args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm,
                   'te', args.te_embedding, args.te_hidden, args.te_layer,
                   'res', args.res_layer,
                   'cf', args.cf_filter, args.fc_hidden, args.fc_layer,
                   args.memo]

    args.config = '_'.join(map(str, config_list))
    args.log = os.path.join(args.log_directory, args.project, args.time_stamp + args.config)
    print("Config:", args.config)

    return args
