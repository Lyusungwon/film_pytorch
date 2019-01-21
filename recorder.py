import torch
import time
from collections import defaultdict
import wandb

class Recorder:
    def __init__(self, writer, args, batch_record_idx=0):
        self.writer = writer
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.idx_to_question_type = args.idx_to_question_type
        self.idx_to_word = args.idx_to_word
        self.answer_idx_to_word = args.answer_idx_to_word
        self.qt_size = args.qt_size
        self.batch_record_idx = batch_record_idx
        self.epoch_idx = None
        self.mode = None
        self.batch_num = 0
        self.dataset_size = 0
        self.epoch_loss = 0
        self.epoch_correct = 0
        self.per_question = None
        self.per_question_type = None
        self.epoch_start_time = 0
        self.epoch_end_time = 0
        self.epoch_time = 0
        self.batch_loss = 0
        self.batch_correct = 0
        self.batch_start_time = 0
        self.batch_end_time = 0
        self.batch_time = 0

    def epoch_start(self, epoch_idx, is_train, loader):
        self.epoch_idx = epoch_idx
        self.mode = 'Train' if is_train else 'Test'
        self.batch_num = len(loader)
        self.dataset_size = len(loader.dataset)
        self.epoch_loss = 0
        self.epoch_correct = 0
        self.per_question = {"correct": defaultdict(lambda: 0),
                             "number": defaultdict(lambda: 0)
                             }
        self.per_question_type = {"correct": defaultdict(lambda: 0),
                                  "number": defaultdict(lambda: 0)
                                  }
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()

    def batch_end(self, loss, correct, types):
        self.batch_end_time = time.time()
        self.batch_loss = loss.item()
        self.epoch_loss += loss.item()
        self.record_types(correct, types)
        self.batch_correct = correct.sum().item()
        self.epoch_correct += correct.sum().item()
        self.batch_time = self.batch_end_time - self.batch_start_time
        self.batch_start_time = time.time()

    def record_types(self, correct, types):
        correct = correct.cpu()
        for question_type in types:
            for i in range(self.qt_size):
                idx = question_type == i
                self.per_question["correct"][i] += (correct * idx).sum().item()
                self.per_question["number"][i] += idx.sum().item()

    def log_batch(self, batch_idx, batch_size):
        print('Train Batch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} / Time: {:.4f} / Acc: {:.4f}'.format(
            self.epoch_idx,
            batch_idx * batch_size, self.dataset_size,
            100. * batch_idx / self.batch_num,
            self.batch_loss / batch_size,
            self.batch_time,
            self.batch_correct / batch_size))
        wandb.log({f"{self.mode} batch loss": self.batch_loss / batch_size,
                   f"{self.mode} batch accuracy": self.batch_correct / batch_size,
                   f"{self.mode} batch time": self.batch_time})
        self.writer.add_scalar('{}-4.Batch loss'.format(self.mode), self.batch_loss / batch_size, self.batch_record_idx)
        self.writer.add_scalar('{}-5.Batch accuracy'.format(self.mode), self.batch_correct / batch_size, self.batch_record_idx)
        self.writer.add_scalar('{}-6.Batch time'.format(self.mode), self.batch_time, self.batch_record_idx)

        self.batch_record_idx += 1

    def log_epoch(self, idx_to_question_type=None):
        if idx_to_question_type:
            self.idx_to_question_type = idx_to_question_type
        self.epoch_end_time = time.time()
        self.epoch_time = self.epoch_end_time - self.epoch_start_time
        print('====> {}: {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
            self.mode,
            self.epoch_idx,
            self.epoch_loss / self.dataset_size,
            self.epoch_time,
            self.epoch_correct / self.dataset_size))
        wandb.log({f"{self.mode} epoch loss": self.epoch_loss / self.dataset_size,
                   f"{self.mode} epoch accuracy": self.epoch_correct / self.dataset_size,
                   f"{self.mode} epoch time": self.epoch_time})
        self.writer.add_scalar('{}-1.Total loss'.format(self.mode), self.epoch_loss / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar('{}-2.Total accuracy'.format(self.mode), self.epoch_correct / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar('{}-3.Total time'.format(self.mode), self.epoch_time, self.epoch_idx)
        per_question_log = dict()
        for question_type_idx, question_type_name in self.idx_to_question_type.items():
            self.per_question_type['correct'][question_type_name] += self.per_question['correct'][question_type_idx]
            self.per_question_type['number'][question_type_name] += self.per_question['number'][question_type_idx]
        for question_type_name in self.per_question_type['correct'].keys():
            type_accuracy = self.per_question_type['correct'][question_type_name] / self.per_question_type['number'][question_type_name]
            per_question_log[f"{self.mode} question {question_type_nema} accuracy"] = type_accuracy
            self.writer.add_scalar("{}-7. Question '{}' accuracy".format(self.mode, question_type_name), type_accuracy, self.epoch_idx)
        wandb.log(per_question_log)

    def log_data(self, image, question, answer):
        n = min(self.batch_size, 4)
        question_text = [' '.join([self.idx_to_word[i] for i in q]) for q in question.cpu().numpy()[:n]]
        answer_text = [self.answer_idx_to_word[a] for a in answer.cpu().numpy()[:n]]
        text = []
        for j, (question, answer) in enumerate(zip(question_text, answer_text)):
            text.append(f'Quesetion {j}: {question} / Answer: {answer}')
        self.writer.add_image('Image', torch.cat([image[:n]]), self.epoch_idx)
        self.writer.add_text('QA', '\n'.join(text), self.epoch_idx)

    def get_epoch_loss(self):
        return self.epoch_loss / self.dataset_size
