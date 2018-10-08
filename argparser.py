import argparse
import datetime
from pathlib import Path
home = str(Path.home())

def default_parser():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='N',
                        help='learning rate (default: 2.5e-4)')
    parser.add_argument('--lr-term', type=int, default=500, metavar='N',
                        help='term of increase learning rate (default: 500)')
    parser.add_argument('--lr-inc', type=float, default=2, metavar='N',
                        help='increase of learning rate (default: 2)')
    parser.add_argument('--data-directory', type=str, default = home + '/data/', metavar='N',
                        help='directory of data')
    parser.add_argument('--log-directory', type=str, default = home + '/experiment/', metavar='N',
                        help='log directory')
    parser.add_argument('--device', type=int, default=0, metavar='N',
                        help='number of cuda')
    parser.add_argument('--cpu-num', type=int, default=0, metavar='N',
                        help='number of cpu')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--time-stamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N',
                        help='time of the run(no modify)')
    parser.add_argument('--memo', type=str, default='default', metavar='N',
                        help='memo of the model')
    parser.add_argument('--load-model', type=str, default='000000000000', metavar='N',
                        help='load previous model')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='start-epoch number')

    return parser
