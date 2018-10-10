import sys
sys.path.append('../utils/')
import argparser
import dataloader
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from tensorboardX import SummaryWriter
import model
from collections import defaultdict
import cv2

parser = argparser.default_parser()
# Input
parser.add_argument('--name', type=str, default='rn')
parser.add_argument('--dataset', type=str, default='sortofclevr2')
parser.add_argument('--channel-size', type=int, default=3)
parser.add_argument('--input-h', type=int, default=128)
parser.add_argument('--input-w', type=int, default=128)
# Convolution
parser.add_argument('--cv-filter', type=int, default=24)
parser.add_argument('--cv-kernel', type=int, default=3)
parser.add_argument('--cv-stride', type=int, default=2)
parser.add_argument('--cv-layer', type=int, default=4)
parser.add_argument('--cv-batchnorm', action='store_true')
# Text Encoder
parser.add_argument('--te-embedding', type=int, default=64)
parser.add_argument('--te-hidden', type=int, default=128)
parser.add_argument('--te-layer', type=int, default=1)
# # g theta
# parser.add_argument('--gt-hidden', type=int, default=256)
# parser.add_argument('--gt-layer', type=int, default=4)
# self attention
parser.add_argument('--sa-nlayer', type=int, default=2)
parser.add_argument('--sa-inner', type=int, default=1024)
parser.add_argument('--sa-dropout', type=int, default=0.1)
parser.add_argument('--sa-nhead', type=int, default=10)
parser.add_argument('--sa-key', type=int, default=256)
parser.add_argument('--sa-value', type=int, default=256)
# f phi
parser.add_argument('--fp-hidden', type=int, default=1024)
parser.add_argument('--fp-dropout', type=int, default=3)
parser.add_argument('--fp-dropout-rate', type=float, default=0.2)
parser.add_argument('--fp-layer', type=int, default=4)

args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.device))
    torch.cuda.set_device(args.device)

config_list = [args.name, args.dataset, args.epochs, args.batch_size, 
                args.lr, args.lr_term, args.lr_inc, args.device,
               'inp', args.channel_size, args.input_h, args.input_w,
               'cv', args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm,
               'te', args.te_embedding, args.te_hidden, args.te_layer,
               'sa', args.sa_nlayer, args.sa_inner, args.sa_dropout, args.sa_nhead, args.sa_key, args.sa_value,
               # 'gt', args.gt_hidden, args.gt_layer,
               'fp', args.fp_hidden, args.fp_dropout, args.fp_dropout_rate, args.fp_layer,
               args.memo]
config = '_'.join(map(str, config_list))
print("Config:", config)

train_loader = dataloader.train_loader(args.dataset, args.data_directory, args.batch_size, args.input_h, args.input_w, args.cpu_num)
test_loader = dataloader.test_loader(args.dataset, args.data_directory, args.batch_size, args.input_h, args.input_w, args.cpu_num)

cv_layout = [(args.cv_filter, args.cv_kernel, args.cv_stride) for i in range(args.cv_layer)]
fp_layout = [64 * ((args.cv_filter + 2) + 2 * args.te_embedding)] + [args.fp_hidden for i in range(args.fp_layer)] + [train_loader.dataset.a_size]

conv = model.Conv(cv_layout, args.channel_size, args.cv_batchnorm).to(device)
f_phi = model.MLP(fp_layout, args.fp_dropout, args.fp_dropout_rate, last=True).to(device)
if args.dataset == 'clevr':
    text_encoder = model.Text_encoder(train_loader.dataset.q_size, args.te_embedding, args.te_hidden, args.te_layer).to(device)
else:
    text_encoder = model.Text_embedding(train_loader.dataset.c_size, train_loader.dataset.q_size, args.te_embedding).to(device)
self_attention = model.SelfAttention(args.sa_nlayer, (args.cv_filter + 2) + 2 * args.te_embedding, args.sa_inner, args.sa_nhead, args.sa_key, args.sa_value, args.sa_dropout).to(device)

if args.load_model != '000000000000':
    conv.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/conv.pt'))
    f_phi.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/    f_phi.pt'))
    text_encoder.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/text_encoder.pt'))
    self_attention.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/self_attention.pt'))
    args.time_stamp = args.load_model[:12]
    print('Model {} loaded.'.format(args.load_model))

log = args.log_directory + args.name + '/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(list(conv.parameters()) + list(self_attention.parameters()) + list(text_encoder.parameters()) + list(f_phi.parameters()), lr=args.lr)


def positional_encoding(images, questions):
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    images = images.view(n, c, o).transpose(1, 2)
    x_coordinate = torch.linspace(-1, 1, h).view(1, h, 1, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    questions = questions.unsqueeze(1).expand(n, o, hd)
    images = torch.cat([images, x_coordinate, y_coordinate, questions], 2)
    return images

def train(epoch):
    epoch_start_time = time.time()
    start_time = time.time()
    train_loss = 0
    train_correct = 0
    batch_num = 0
    batch_loss = 0
    batch_correct = 0
    f_phi.train()
    conv.train()
    text_encoder.train()
    if (epoch + 1) % args.lr_term == 0:
        lr = 0
        for params in optimizer.param_groups:
            lr = min(0.001, params['lr'] * args.lr_inc)
            params['lr'] = lr
        print("Learning rate updated to {}".format(lr))
    for batch_idx, (image, question, answer) in enumerate(train_loader):
        batch_size = image.size()[0]
        optimizer.zero_grad()
        image = image.to(device)
        answer = answer.to(device)
        objects = conv(image)
        if args.dataset == 'clevr':
            question = PackedSequence(question.data.to(device), question.batch_sizes)
        else:
            question = question.to(device)
            # answer = answer.squeeze(1)
        questions = text_encoder(question)
        encoded_objects = positional_encoding(objects, questions)
        attended_objects = self_attention(encoded_objects)
        output = f_phi(attended_objects)
        loss = F.cross_entropy(output, answer)
        loss.backward()
        optimizer.step()
        pred = torch.max(output.data, 1)[1]
        correct = (pred == answer).sum()
        train_loss += loss.item()
        train_correct += correct.item()
        batch_num += batch_size
        batch_loss += loss.item()
        batch_correct += correct.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} / Time: {:.4f} / Acc: {:.4f}'.format(
                epoch,
                batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                batch_loss / batch_num,
                time.time() - start_time,
                batch_correct / batch_num))
            idx = epoch * len(train_loader) // args.log_interval + batch_idx // args.log_interval
            writer.add_scalar('Batch loss', batch_loss / batch_num, idx)
            writer.add_scalar('Batch accuracy', batch_correct / batch_num, idx)
            writer.add_scalar('Batch time', time.time() - start_time, idx)
            batch_num = 0
            batch_loss = 0
            batch_correct = 0
            start_time = time.time()

    print('====> Epoch: {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
        epoch,
        train_loss / len(train_loader.dataset),
        time.time() - epoch_start_time,
        train_correct / len(train_loader.dataset)))
    writer.add_scalar('Train loss', train_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('Train accuracy', train_correct / len(train_loader.dataset), epoch)


def test(epoch):
    f_phi.eval()
    conv.eval()
    text_encoder.eval()
    test_loss = 0
    q_correct = defaultdict(lambda: 0)
    q_num = defaultdict(lambda: 0)
    for batch_idx, (image, question, answer) in enumerate(test_loader):
        batch_size = image.size()[0]
        image = image.to(device)
        answer = answer.to(device)
        objects = conv(image)
        if args.dataset == 'clevr':
            question = PackedSequence(question.data.to(device), question.batch_sizes)
        else:
            question = question.to(device)
            # answer = answer.squeeze(1)
        questions = text_encoder(question)
        encoded_objects = positional_encoding(objects, questions)
        attended_objects = self_attention(encoded_objects)
        output = f_phi(attended_objects)
        loss = F.cross_entropy(output, answer)
        test_loss += loss.item()
        pred = torch.max(output.data, 1)[1]
        correct = (pred == answer)
        for i in range(6):
            idx = question[:, 1] == i
            q_correct[i] += (correct * idx).sum().item()
            q_num[i] += idx.sum().item()
        if batch_idx == 0:
            n = min(batch_size, 4)
            if args.dataset == 'clevr':
                pad_question, lengths = pad_packed_sequence(question)
                pad_question = pad_question.transpose(0, 1)
                question_text = [' '.join([train_loader.dataset.idx_to_word[i] for i in q]) for q in pad_question.cpu().numpy()[:n]]
                answer_text = [train_loader.dataset.answer_idx_to_word[a] for a in answer.cpu().numpy()[:n]]
                text = []
                for j, (q, a) in enumerate(zip(question_text, answer_text)):
                    text.append('Quesetion {}: '.format(j) + question_text[j] + '/ Answer: ' + answer_text[j])
                writer.add_image('Image', torch.cat([image[:n]]), epoch)
                writer.add_text('QA', '\n'.join(text), epoch)
            else:
                image = F.pad(image[:n], (0, 0, 0, 20), mode='constant', value=1).transpose(1,2).transpose(2,3)
                image = image.cpu().numpy()
                for i in range(n):
                    cv2.line(image[i], (64, 0), (64, 128), (0, 0, 0), 1)
                    cv2.line(image[i], (0, 64), (128, 64), (0, 0, 0), 1)
                    cv2.line(image[i], (0, 128), (128, 128), (0, 0, 0), 1)
                    cv2.putText(image[i], '{} {} {} {}'.format(
                        train_loader.dataset.idx_to_color[question[i, 0].item()],
                        train_loader.dataset.idx_to_question[question[i, 1].item()],
                        train_loader.dataset.idx_to_answer[answer[i].item()],
                        train_loader.dataset.idx_to_answer[pred[i].item()]),
                        (2, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                image = torch.from_numpy(image).transpose(2,3).transpose(1,2)
                writer.add_image('Image', torch.cat([image]), epoch)
    print('====> Test set loss: {:.4f}\tAccuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset), sum(q_correct.values())/len(test_loader.dataset)))
    writer.add_scalar('Test loss', test_loss / len(test_loader.dataset), epoch)
    q_acc = {}
    for i in range(6):
        q_acc['question {}'.format(str(i))] = q_correct[i]/q_num[i]
    writer.add_scalars('Test accuracy per question', q_acc, epoch)
    writer.add_scalar('Test total accuracy', sum(q_correct.values())/len(test_loader.dataset), epoch)

for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    torch.save(conv.state_dict(), log + 'conv.pt')
    torch.save(text_encoder.state_dict(), log + 'text_encoder.pt')
    torch.save(f_phi.state_dict(), log + 'f_phi.pt')
    torch.save(self_attention.state_dict(), log + 'self_attention.pt')
    print('Model saved in ', log)
writer.close()
