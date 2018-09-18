import sys
sys.path.append('../utils/')
import argparser
import dataloader
import model
import time
import os
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from tensorboardX import SummaryWriter
from model import MLP, Conv, Text_encoder
import numpy as np
import pickle
import cProfile

parser = argparser.default_parser()
# Input
parser.add_argument('--name', type=str, default='rn')
parser.add_argument('--channel-size', type=int, default=3)
parser.add_argument('--input-h', type=list, default=128)
parser.add_argument('--input-w', type=list, default=128)
# Convolution
parser.add_argument('--cv-filter', type=int, default=24)

parser.add_argument('--cv-kernel', type=int, default=3)
parser.add_argument('--cv-stride', type=int, default=2)
parser.add_argument('--cv-layer', type=int, default=4)
# Text Encoder
parser.add_argument('--te-embedding', type=int, default=32)
parser.add_argument('--te-hidden', type=int, default=128)
parser.add_argument('--te-layer', type=int, default=1)
# g theta
parser.add_argument('--gt-hidden', type=int, default=256)
parser.add_argument('--gt-layer', type=int, default=4)
# f phi
parser.add_argument('--fp-hidden', type=int, default=256)
parser.add_argument('--fp-dropout', type=int, default=2)
parser.add_argument('--fp-dropout_rate', type=float, default=0.5)
parser.add_argument('--fp-layer', type=int, default=3)

args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.device))
    torch.cuda.set_device(args.device)

config_list = [args.epochs, args.batch_size, args.lr, args.device,
               'inp', args.channel_size, args.input_h, args.input_w,
               'cv', args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer,
               'te', args.te_embedding, args.te_hidden, args.te_layer,
               'gt', args.gt_hidden, args.gt_layer,
               'fp', args.fp_hidden, args.fp_dropout, args.fp_dropout_rate, args.fp_layer]
config = '_'.join(map(str, config_list))
print("Config:", config)

train_loader = dataloader.train_loader('clevr', args.data_directory, args.batch_size, args.input_h, args.input_w, args.cpu_num)
test_loader = dataloader.test_loader('clevr', args.data_directory, args.batch_size, args.input_h, args.input_w, args.cpu_num)

cv_layout = [(args.cv_filter, args.cv_kernel, args.cv_stride) for i in range(args.cv_layer)]
gt_layout = [args.gt_hidden for i in range(args.gt_layer)]
gt_layout.insert(0, (args.cv_filter + 1) * 2 + args.te_hidden)
fp_layout = [args.fp_hidden for i in range(args.fp_layer)]
fp_layout.append(train_loader.dataset.a_size)

conv = Conv(cv_layout, args.channel_size).to(device)
text_encoder = Text_encoder(train_loader.dataset.q_size, args.te_embedding, args.te_hidden, args.te_layer).to(device)
g_theta = MLP(gt_layout).to(device)
f_phi = MLP(fp_layout, args.fp_dropout, args.fp_dropout_rate, last = True).to(device)
with open('data_dict.pkl', 'rb') as file:
    data_dict = pickle.load(file)

if args.load_model != '000000000000':
    conv.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/conv.pt'))
    text_encoder.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/text_encoder.pt'))
    gt.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/gt.pt'))
    fp.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/fp.pt'))
    args.time_stamep = args.load_mode[:12]

log = args.log_directory + args.name + '/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(text_encoder.parameters(), lr=args.lr)

def object_pair(images, questions):
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    cordinate = torch.linspace(-1, 1, o).view(1, o, 1).expand(n, o, 1).to(device)
    images = images.view(n, c, o).transpose(1, 2)
    images = torch.cat([images, cordinate], 2)
    images1 = images.unsqueeze(1).expand(n, o, o, c + 1).contiguous().view(n, o**2, c + 1)
    images2 = images.unsqueeze(2).expand(n, o, o, c + 1).contiguous().view(n, o**2, c + 1)
    questions = questions.unsqueeze(1).expand(n, o**2, hd)
    pairs = torch.cat([images1, images2, questions], 2)
    # pairs = pairs.view(n * (o**2), -1)
    return pairs

def train(epoch):
    epoch_start_time = time.time()
    start_time = time.time()
    train_loss = 0
    train_correct = 0
    batch_num = 0
    batch_loss = 0
    batch_correct = 0
    g_theta.train()
    f_phi.train()
    conv.train()
    text_encoder.train()
    for batch_idx, (image, question, answer) in enumerate(train_loader):
        # a = time.time() - start_time
        # print("load", a)
        # start_time = time.time()
        batch_size = image.size()[0]
        optimizer.zero_grad()
        image = image.to(device)
        question= PackedSequence(question.data.to(device), question.batch_sizes)
        answer = answer.to(device)
        objects = conv(image)
        questions = text_encoder(question)
        pairs = object_pair(objects, questions)
        relations = g_theta(pairs)
        relations_sum = relations.sum(1)
        output = f_phi(relations_sum)
        loss = F.cross_entropy(output, answer)
        loss.backward()
        optimizer.step()

        correct = (torch.max(output.data, 1)[1] == answer).sum()
        train_loss += loss.item()
        train_correct += correct.item()
        batch_num += batch_size
        batch_loss += loss.item()
        batch_correct += correct.item()
        # a = time.time() - start_time
        # print('cal', a)
        # start_time = time.time()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] / Loss: {:.4f} / Time: {:.4f} / Acc: {:.4f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), 
                batch_loss / batch_num, 
                time.time() - start_time,
                batch_correct / batch_num))

            start_time = time.time()
            idx = epoch * len(train_loader) // args.log_interval + batch_idx // args.log_interval
            writer.add_scalar('Train batch loss',  batch_loss / batch_num, idx) 
            writer.add_scalar('Train batch accuracy',  batch_correct / batch_num, idx) 
            batch_num = 0
            batch_loss = 0
            batch_correct = 0

    print('====> Epoch: {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
        epoch, 
        train_loss / len(train_loader.dataset), 
        time.time() - epoch_start_time,
        train_correct / len(train_loader.dataset)))
    writer.add_scalar('Train loss',  train_loss / len(train_loader.dataset), epoch) 
    writer.add_scalar('Train accuracy',  1. * train_correct / len(train_loader.dataset), epoch) 


def test(epoch):
    g_theta.eval()
    f_phi.eval()
    conv.eval()
    text_encoder.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (image, question, answer) in enumerate(train_loader):
        start_time = time.time()
        batch_size = image.size()[0]
        optimizer.zero_grad()
        image = image.to(device)
        question= PackedSequence(question.data.to(device), question.batch_sizes)
        answer = answer.to(device)
        objects = conv(image)
        questions = text_encoder(question)
        pairs = object_pair(objects, questions)
        relations = g_theta(pairs)
        relations_sum = relations.view(batch_size, -1, args.gt_hidden).sum(1)
        output = f_phi(relations_sum)
        loss = F.cross_entropy(output, answer)
        test_loss += loss.item()
        correct += (torch.max(output.data, 1)[1] == answer).sum().item()

        if batch_idx == 0:
            n = min(batch_size, 4)
            pad_question, lengths = pad_packed_sequence(question)
            pad_question = pad_question.transpose(0,1)
            question_text = [' '.join([data_dict['question']['idx_to_word'][i] for i in q]) for q in pad_question.cpu().numpy()[:n]]
            answer_text = [data_dict['answer']['idx_to_word'][a] for a in answer.cpu().numpy()[:n]]
            text = []
            for j, (q, a) in enumerate(zip(question_text, answer_text)):
                text.append('Quesetion {}: '.format(j) + question_text[j] + '/ Answer: ' + answer_text[j])
            writer.add_image('Image', torch.cat([image[:n]]), epoch)
            writer.add_text('QA', '\n'.join(text), epoch)
    print('====> Test set loss: {:.4f}\tAccuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset), test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
    writer.add_scalar('Test loss',  test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test accuracy',  correct / len(test_loader.dataset), epoch)

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    torch.save(g_theta, log + 'g_theta.pt')
    torch.save(f_phi, log + 'f_phi.pt')
    torch.save(conv, log + 'conv.pt')
    torch.save(text_encoder, log + 'text_encoder.pt')
    print('Model saved in ', log)
writer.close()
