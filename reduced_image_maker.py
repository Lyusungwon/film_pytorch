import os
import torch
import dataloader
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
from utils import load_pretrained_conv
home = str(Path.home())

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--data-directory', type=str, default=os.path.join(home, 'data'), metavar='N',
                    help='directory of data')
parser.add_argument('--dataset', type=str, default='clevr')
parser.add_argument('--input-h', type=int, default=128)
parser.add_argument('--input-w', type=int, default=128)
parser.add_argument('--cpu-num', type=int, default=1)
parser.add_argument('--is-train', action='store_true')
parser.add_argument('--device', type=int, default=0, metavar='N', help='gpu number')
args, unparsed = parser.parse_known_args()
args.data_config = [args.input_h, args.input_w, args.cpu_num]
if not torch.cuda.is_available():
    args.device = torch.device('cpu')
else:
    torch.cuda.set_device(args.device)
    args.device = torch.device(args.device)


def make_reduced_images():
    mode = 'train' if args.is_train else 'test'
    feature_extractor = load_pretrained_conv().to(args.device)
    loader = dataloader.load_dataloader(args.dataset, args.data_directory, args.is_train, 1, args.data_config)
    reduced_images = []
    for image, _, _, _ in tqdm(loader):
        image = image.to(args.device)
        reduced_image = feature_extractor(image)
        reduced_images.append(reduced_image)
    # reduced_images = torch.cat(reduced_images, 0)
    with open(os.path.join(args.data_directory, args.dataset, f'{mode}_reduced_images.pkl'), 'wb') as file:
        pickle.dump(reduced_images, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{mode}_reduced_images.pkl saved')


if __name__ =='__main__':
    make_reduced_images()
