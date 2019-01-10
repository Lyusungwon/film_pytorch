import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import *
from configuration import get_config
from film import Film
from san import San
import dataloader
from collections import defaultdict

args = get_config()
device = args.device
torch.manual_seed(args.seed)

train_loader = dataloader.load_dataloader(args.dataset, args.data_directory, True, args.batch_size, args.data_config)
test_loader = dataloader.load_dataloader(args.dataset, args.data_directory, False, args.batch_size, args.data_config)
args.a_size = train_loader.dataset.a_size
args.q_size = train_loader.dataset.q_size
args.qt_size = 5
if args.te_pretrained:
    args.word2idx = train_loader.dataset.word_to_idx

if args.model == 'film':
    model = Film(args)
elif args.model == 'san':
    model = San(args)

if args.multi_gpu:
    model = nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)])
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.load_model:
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.log)
model = model.to(device)
start_epoch = 0


def epoch(epoch_idx, is_train):
    epoch_start_time = time.time()
    start_time = time.time()
    mode = 'Train' if is_train else 'Test'
    epoch_loss = 0
    total_correct = 0
    q_correct = defaultdict(lambda: 0)
    q_num = defaultdict(lambda: 0)
    model.train() if is_train else model.eval()
    loader = train_loader if is_train else test_loader
    for batch_idx, (image, question_set, answer, question_type) in enumerate(loader):
        batch_size = image.size()[0]
        optimizer.zero_grad()
        image = image.to(device)
        question = question_set[0].to(device)
        question_length = question_set[1].to(device)
        answer = answer.to(device)
        question_type = question_type.to(device)
        output = model(image * 2 - 1, question, question_length)
        loss = F.cross_entropy(output, answer)
        if is_train:
            loss.backward()
            optimizer.step()
        batch_loss = loss.item()
        epoch_loss += batch_loss
        pred = torch.max(output.data, 1)[1]
        correct = (pred == answer)
        batch_correct = correct.sum().item()
        total_correct += batch_correct
        for i in range(args.qt_size):
            idx = question_type//10 - 1 == i
            q_correct[i] += (correct * idx).sum().item()
            q_num[i] += idx.sum().item()
        if is_train:
            if batch_idx % args.log_interval == 0:
                print('Train Batch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} / Time: {:.4f} / Acc: {:.4f}'.format(
                    epoch_idx,
                    batch_idx * batch_size, len(loader.dataset),
                    100. * batch_idx / len(loader),
                    batch_loss / batch_size,
                    time.time() - start_time,
                    batch_correct / batch_size))
                idx = epoch_idx * len(loader) // args.log_interval + batch_idx // args.log_interval
                writer.add_scalar('Batch loss', batch_loss / batch_size, idx)
                writer.add_scalar('Batch accuracy', batch_correct / batch_size, idx)
                writer.add_scalar('Batch time', time.time() - start_time, idx)
                start_time = time.time()
        # else:
        #     if batch_idx == 0:
        #         n = min(batch_size, 4)
        #         pad_question, lengths = pad_packed_sequence(question)
        #         pad_question = pad_question.transpose(0, 1)
        #         question_text = [' '.join([loader.dataset.idx_to_word[i] for i in q]) for q in
        #                          pad_question.cpu().numpy()[:n]]
        #         answer_text = [loader.dataset.answer_idx_to_word[a] for a in answer.cpu().numpy()[:n]]
        #         text = []
        #         for j, (q, a) in enumerate(zip(question_text, answer_text)):
        #             text.append('Quesetion {}: '.format(j) + question_text[j] + '/ Answer: ' + answer_text[j])
        #         writer.add_image('Image', torch.cat([image[:n]]), epoch_idx)
        #         writer.add_text('QA', '\n'.join(text), epoch_idx)

    print('====> {}: {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
        mode,
        epoch_idx,
        epoch_loss / len(loader.dataset),
        time.time() - epoch_start_time,
        total_correct / len(loader.dataset)))
    writer.add_scalar('{} loss'.format(mode), epoch_loss / len(loader.dataset), epoch_idx)
    writer.add_scalar('{} total accuracy'.format(mode), total_correct / len(loader.dataset), epoch_idx)
    for i in range(args.qt_size):
        writer.add_scalar('{} accuracy for question {}'.format(mode, i), q_correct[i] / q_num[i], epoch_idx)
    # q_corrects = list(q_correct.values())
    # q_nums = list(q_num.values())
    # writer.add_scalar('{} non-rel accuracy'.format(mode), sum(q_corrects[:3]) / sum(q_nums[:3]), epoch_idx)
    # writer.add_scalar('{} rel accuracy'.format(mode), sum(q_corrects[3:]) / sum(q_nums[3:]), epoch_idx)
    # writer.add_scalar('{} total accuracy'.format(mode), sum(q_correct.values()) / len(loader.dataset), epoch_idx)


if __name__ == '__main__':
    writer = SummaryWriter(args.log)
    for epoch_idx in range(start_epoch, args.epochs):
        epoch(epoch_idx, is_train=True)
        epoch(epoch_idx, is_train=False)
        save_checkpoint(epoch_idx, model.module if args.multi_gpu else model, optimizer, args.log)
    writer.close()
