import time
import torch.optim as optim
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from tensorboardX import SummaryWriter
import model
from utils import *
from collections import defaultdict
import cv2
from configuration import get_config
import numpy as np
import dataloader

args = get_config()
device = args.device
torch.manual_seed(args.seed)

train_loader = dataloader.train_loader(args.dataset, args.data_directory, args.batch_size, args.data_config)
test_loader = dataloader.test_loader(args.dataset, args.data_directory, args.batch_size, args.data_config)

models = dict()
cv_layout = [(args.cv_filter, args.cv_kernel, args.cv_stride) for i in range(args.cv_layer)]
if args.model == 'baseline':
    gt_layout = [(args.cv_filter + 2) + args.te_embedding * 2] + [args.gt_hidden for i in range(args.gt_layer)]
    fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 1)] + [train_loader.dataset.a_size]
    conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
    g_theta = model.MLP(gt_layout).to(device)
    f_phi = model.MLP(fp_layout).to(device)
    models['g_theta.pt'] = g_theta
    models['f_phi.pt'] = f_phi

elif args.model == 'rn':
    gt_layout = [(args.cv_filter + 2) * 2 + args.te_embedding * 2] + [args.gt_hidden for i in range(args.gt_layer)]
    fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 1)] + [train_loader.dataset.a_size]
    conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
    g_theta = model.MLP(gt_layout).to(device)
    f_phi = model.MLP(fp_layout).to(device)
    models['g_theta.pt'] = g_theta
    models['f_phi.pt'] = f_phi

elif args.model == 'sarn':
    gt_layout = [2 * (args.cv_filter + 2 + args.te_embedding)] + [args.gt_hidden for i in range(args.gt_layer)]
    hp_layout = [args.cv_filter + 2 + args.te_embedding * 2] + [args.hp_hidden for i in range(args.hp_layer - 1)] + [1]
    fp_layout = [args.gt_hidden] + [args.fp_hidden for i in range(args.fp_layer - 1)] + [train_loader.dataset.a_size]
    conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
    g_theta = model.MLP(gt_layout).to(device)
    h_psi = model.MLP(hp_layout).to(device)
    f_phi = model.MLP(fp_layout).to(device)
    models['g_theta.pt'] = g_theta
    models['f_phi.pt'] = f_phi
    models['h_psi.pt'] = h_psi

elif args.model == 'sarn_att':
    gt_layout = [2 * (args.cv_filter + 2 + args.te_embedding)] + [args.gt_hidden for i in range(args.gt_layer)]
    hp_layout = [args.cv_filter + 2 + args.te_embedding * 2] + [args.hp_hidden for i in range(args.hp_layer - 1)] + [1]
    fp_layout = [args.cv_filter + 2] + [args.fp_hidden for i in range(args.fp_layer - 2)] + [train_loader.dataset.a_size]
    h_psi = model.MLP(hp_layout).to(device)
    conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
    g_theta = model.MLP(gt_layout).to(device)
    h_psi = model.MLP(hp_layout).to(device)
    attn = model.MultiHeadAttention(n_head=1, d_model=args.cv_filter+2, d_k=32, d_v=32).to(device)
    f_phi = model.MLP(fp_layout).to(device)
    models['g_theta.pt'] = g_theta
    models['f_phi.pt'] = f_phi
    models['h_psi.pt'] = h_psi
    models['attn.pt'] = attn

elif args.model == 'new':
    fp_layout = [args.fp_hidden for i in range(args.fp_layer - 1)] + [train_loader.dataset.a_size]
    conv = model.Filmm(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm, args.chaining_size, args.lstm_hidden, args.te_embedding * 2, args.fp_hidden).to(device)
    f_phi = model.MLP(fp_layout).to(device)
    models['f_phi.pt'] = f_phi

elif args.model == 'film':
    conv = model.Conv(args.input_h, args.input_w, cv_layout, args.channel_size, args.cv_layernorm).to(device)
    input_h, input_w = args.input_h, args.input_w
    for i in range(args.cv_layer):
        input_h = int(np.ceil(input_h / 2))
        input_w = int(np.ceil(input_w / 2))
    film = model.Film(args.te_embedding * 2, args.film_lstm_hidden, args.film_filter, args.film_kernel, args.film_res_layer, args.film_last_filter, input_h, input_w, args.film_mlp_hidden, args.film_mlp_layer, train_loader.dataset.a_size).to(device)
    models['film.pt'] = film

if args.dataset == 'clevr':
    text_encoder = model.Text_encoder(train_loader.dataset.q_size, args.te_embedding, args.te_hidden, args.te_layer).to(device)
else:
    text_encoder = model.Text_embedding(train_loader.dataset.c_size, train_loader.dataset.q_size, args.te_embedding).to(device)
models['text_encoder.pt'] = text_encoder
models['conv.pt'] = conv


if args.load_model != '000000000000':
    for model_name, model in models.items():
        model.load_state_dict(torch.load(args.log_directory + args.project + '/' + args.load_model + '/' + model_name))
    args.time_stamp = args.load_model[:12]
    print('Model {} loaded.'.format(args.load_model))


def epoch(epoch_idx, is_train):
    epoch_start_time = time.time()
    start_time = time.time()
    mode = 'Train' if is_train else 'Test'
    epoch_loss = 0
    q_correct = defaultdict(lambda: 0)
    q_num = defaultdict(lambda: 0)
    if is_train:
        for model in models.values():
            model.train()
        loader = train_loader
    else:
        for model in models.values():
            model.eval()
        loader = test_loader
    for batch_idx, (image, question, answer) in enumerate(loader):
        batch_size = image.size()[0]
        optimizer.zero_grad()
        image = image.to(device)
        answer = answer.to(device)
        if args.dataset == 'clevr':
            question = PackedSequence(question.data.to(device), question.batch_sizes)
        else:
            question = question.to(device)
            # answer = answer.squeeze(1)
        code = text_encoder(question)
        if args.model == 'baseline':
            objects = conv(image * 2 - 1)
            pairs = baseline_encode(objects, code)
            relations = g_theta(pairs)
            relations = relations.sum(1)
            output = f_phi(relations)
        elif args.model == 'rn':
            objects = conv(image * 2 - 1)
            pairs = rn_encode(objects, code)
            relations = g_theta(pairs)
            relations = lower_sum(relations)
            relations = relations.sum(1)
            output = f_phi(relations)
        elif args.model == 'sarn':
            objects = conv(image * 2 - 1)
            coordinate_encoded, question_encoded = sarn_encode(objects, code)
            logits = h_psi(question_encoded)
            pairs = sarn_pair(coordinate_encoded, question_encoded, logits)
            relations = g_theta(pairs)
            relations = relations.sum(1)
            output = f_phi(relations)
        elif args.model == 'sarn_att':
            objects = conv(image * 2 - 1)
            coordinate_encoded, question_encoded = sarn_encode(objects, code)
            logits = h_psi(question_encoded)
            selected = sarn_select(coordinate_encoded, logits)
            relations, att = attn(selected, coordinate_encoded, coordinate_encoded)
            relations = relations.sum(1)
            output = f_phi(relations)
        elif args.model == 'new':
            relations = conv(image * 2 - 1, code)
            output = f_phi(relations)
        elif args.model == 'film':
            objects = conv(image * 2 - 1)
            output = film(objects, code)
        loss = F.cross_entropy(output, answer)
        if is_train:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        pred = torch.max(output.data, 1)[1]
        correct = (pred == answer)
        for i in range(loader.dataset.q_size):
            idx = question[:, 1] == i
            q_correct[i] += (correct * idx).sum().item()
            q_num[i] += idx.sum().item()
        if is_train:
            if batch_idx % args.log_interval == 0:
                print('Train Batch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} / Time: {:.4f} / Acc: {:.4f}'.format(
                    epoch_idx,
                    batch_idx * batch_size, len(loader.dataset),
                    100. * batch_idx / len(loader),
                    loss.item() / batch_size,
                    time.time() - start_time,
                    correct.sum().item() / batch_size))
                idx = epoch_idx * len(loader) // args.log_interval + batch_idx // args.log_interval
                writer.add_scalar('Batch loss', loss.item() / batch_size, idx)
                writer.add_scalar('Batch accuracy', correct.sum().item() / batch_size, idx)
                writer.add_scalar('Batch time', time.time() - start_time, idx)
                start_time = time.time()
        else:
            if batch_idx == 0:
                n = min(batch_size, 4)
                if args.dataset == 'clevr':
                    pad_question, lengths = pad_packed_sequence(question)
                    pad_question = pad_question.transpose(0, 1)
                    question_text = [' '.join([loader.dataset.idx_to_word[i] for i in q]) for q in
                                     pad_question.cpu().numpy()[:n]]
                    answer_text = [loader.dataset.answer_idx_to_word[a] for a in answer.cpu().numpy()[:n]]
                    text = []
                    for j, (q, a) in enumerate(zip(question_text, answer_text)):
                        text.append('Quesetion {}: '.format(j) + question_text[j] + '/ Answer: ' + answer_text[j])
                    writer.add_image('Image', torch.cat([image[:n]]), epoch_idx)
                    writer.add_text('QA', '\n'.join(text), epoch_idx)
                else:
                    image = F.pad(image[:n], (0, 0, 0, args.input_h // 3), mode='constant', value=1).transpose(1,
                                                                                                               2).transpose(
                        2, 3)
                    image = image.cpu().numpy()
                    for i in range(n):
                        cv2.line(image[i], (args.input_w // 2, 0), (args.input_w // 2, args.input_h), (0, 0, 0), 1)
                        cv2.line(image[i], (0, args.input_h // 2), (args.input_w, args.input_h // 2), (0, 0, 0), 1)
                        cv2.line(image[i], (0, args.input_h), (args.input_w, args.input_h), (0, 0, 0), 1)
                        cv2.putText(image[i], '{} {} {} {}'.format(
                            loader.dataset.idx_to_color[question[i, 0].item()],
                            loader.dataset.idx_to_question[question[i, 1].item()],
                            loader.dataset.idx_to_answer[answer[i].item()],
                            loader.dataset.idx_to_answer[pred[i].item()]),
                                    (2, args.input_h + args.input_h // 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    image = torch.from_numpy(image).transpose(2, 3).transpose(1, 2)
                    writer.add_image('Image', torch.cat([image]), epoch_idx)

    print('====> {}: {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
        mode,
        epoch_idx,
        epoch_loss / len(loader.dataset),
        time.time() - epoch_start_time,
        sum(q_correct.values()) / len(loader.dataset)))
    writer.add_scalar('{} loss'.format(mode), epoch_loss / len(loader.dataset), epoch_idx)
    q_acc = {}
    for i in range(loader.dataset.q_size):
        q_acc['question {}'.format(str(i))] = q_correct[i] / q_num[i]
    q_corrects = list(q_correct.values())
    q_nums = list(q_num.values())
    writer.add_scalars('{} accuracy per question'.format(mode), q_acc, epoch_idx)
    writer.add_scalar('{} non-rel accuracy'.format(mode), sum(q_corrects[:3]) / sum(q_nums[:3]), epoch_idx)
    writer.add_scalar('{} rel accuracy'.format(mode), sum(q_corrects[3:]) / sum(q_nums[3:]), epoch_idx)
    writer.add_scalar('{} total accuracy'.format(mode), sum(q_correct.values()) / len(loader.dataset), epoch_idx)


if __name__ == '__main__':
    optimizer = optim.Adam([param for model in models.values() for param in list(model.parameters())], lr=args.lr)
    writer = SummaryWriter(args.log)
    for epoch_idx in range(args.start_epoch, args.start_epoch + args.epochs):
        epoch(epoch_idx, True)
        epoch(epoch_idx, False)
        for model_name, model in models.items():
            torch.save(model.state_dict(), args.log + model_name)
        print('Model saved in ', args.log)
    writer.close()
