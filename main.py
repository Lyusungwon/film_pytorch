import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import *
from configuration import get_config
from film import Film
from san import San
from recorder import Recorder
import dataloader

args = get_config()
device = args.device
torch.manual_seed(args.seed)

train_loader = dataloader.load_dataloader(args.dataset, args.data_directory, True, args.batch_size, args.data_config)
test_loader = dataloader.load_dataloader(args.dataset, args.data_directory, False, args.batch_size, args.data_config)
start_epoch = 0
batch_record_idx = 0

if args.model == 'film':
    model = Film(args)
elif args.model == 'san':
    model = San(args)

if args.multi_gpu:
    model = nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)])
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.load_model:
    model, optimizer, start_epoch, batch_record_idx = load_checkpoint(model, optimizer, args.log, device)
model = model.to(device)


def epoch(epoch_idx, is_train):
    model.train() if is_train else model.eval()
    loader = train_loader if is_train else test_loader
    recorder.epoch_start(epoch_idx, is_train, loader)
    for batch_idx, (image, question_set, answer, types) in enumerate(loader):
        batch_size = image.size()[0]
        optimizer.zero_grad()
        image = image.to(device)
        question = question_set[0].to(device)
        question_length = question_set[1].to(device)
        answer = answer.to(device)
        # types =
        output = model(image * 2 - 1, question, question_length)
        loss = F.cross_entropy(output, answer)
        if is_train:
            loss.backward()
            optimizer.step()
        pred = torch.max(output.data, 1)[1]
        correct = (pred == answer)
        recorder.batch_end(loss.item(), correct, types)
        if is_train and (batch_idx % args.log_interval == 0):
            recorder.log_batch(batch_idx, batch_size)
    recorder.epoch_end()
        # else:
            # recorder.log_data(image.data, question.data)

if __name__ == '__main__':
    writer = SummaryWriter(args.log)
    recorder = Recorder(writer, args, batch_record_idx)
    for epoch_idx in range(start_epoch, args.epochs):
        epoch(epoch_idx, is_train=True)
        epoch(epoch_idx, is_train=False)
        save_checkpoint(epoch_idx, model.module if args.multi_gpu else model, optimizer, args.log, recorder.batch_record_idx)
    writer.close()
