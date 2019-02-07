import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import *
from configuration import get_config
from recorder import Recorder


args, model, train_loader, test_loader = get_config()
device = args.device
torch.manual_seed(args.seed)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.lr_reduce:
    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

start_epoch = 0
batch_record_idx = 0
if args.load_model:
    model, optimizer, start_epoch, batch_record_idx = load_checkpoint(model, optimizer, args.log, device)

criterion = F.binary_cross_entropy_with_logits if args.multi_label else F.cross_entropy
model = nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)]) if args.multi_gpu else model
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
        output = model(image, question, question_length)
        loss = criterion(output, answer)
        if is_train:
            loss.backward()
            if args.gradient_clipping:
                nn.utils.clip_grad_value_(model.parameters(), args.gradient_clipping)
            optimizer.step()
        recorder.batch_end(loss.item(), output.cpu().detach(), answer.cpu(), types.cpu())
        if is_train and (batch_idx % args.log_interval == 0):
            recorder.log_batch(batch_idx, batch_size)
    recorder.log_epoch()
    if not is_train:
        if args.lr_reduce:
            reduce_scheduler.step(recorder.get_epoch_loss())
        if not args.cv_pretrained:
            recorder.log_data(image, question, answer, types)


if __name__ == '__main__':
    writer = SummaryWriter(args.log)
    recorder = Recorder(writer, args, batch_record_idx)
    for epoch_idx in range(start_epoch, args.epochs):
        epoch(epoch_idx, is_train=True)
        epoch(epoch_idx, is_train=False)
        save_checkpoint(epoch_idx, model.module if args.multi_gpu else model, optimizer, args, recorder.batch_record_idx)
    recorder.finish()
    writer.close()
