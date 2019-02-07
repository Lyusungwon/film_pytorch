import os
import csv
import time
import torch
from collections import defaultdict, deque, OrderedDict
from torchvision.utils import make_grid
from utils import is_file_exist
from tempfile import NamedTemporaryFile
import shutil


class Recorder:
    def __init__(self, writer, args, batch_record_idx=0):
        self.writer = writer
        self.args = args
        self.timestamp = args.timestamp
        self.idx_to_question_type = args.idx_to_question_type
        self.idx_to_word = args.idx_to_word
        self.answer_idx_to_word = args.answer_idx_to_word
        self.qt_size = args.qt_size
        self.multi_label = args.multi_label
        self.batch_record_idx = batch_record_idx
        self.csv_file = os.path.join(args.log_directory, args.project, f"{args.project}_log.csv")
        self.rolling_average = 5
        self.logs = defaultdict(lambda: deque(maxlen=self.rolling_average))
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
        self.per_question_log = dict()
        self.exclude = ['data_directory', 'log_directory', 'data_config', 'config', 'log', 'word_to_idx', 'idx_to_word',
                       'answer_word_to_idx', 'answer_idx_to_word', 'question_type_to_idx', 'idx_to_question_type', 'q_size', 'a_size', 'qt_size']
        self.header = self.make_header()
        if not is_file_exist(self.csv_file):
            self.make_csv()
        if not self.is_record_exist():
            self.make_record()

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

    def batch_end(self, loss, output, answer, types):
        if self.multi_label:
            maxi = torch.max(output, 1)[1]
            pred = torch.zeros_like(output.detach())
            pred[torch.arange(output.size()[0]), maxi] = 1.0
            correct = (pred * answer).sum(1)
        else:
            pred = torch.max(output, 1)[1]
            correct = (pred == answer).float()
        self.batch_end_time = time.time()
        self.batch_loss = loss
        self.epoch_loss += self.batch_loss
        self.record_types(correct, types)
        self.batch_correct = correct.sum().item()
        self.epoch_correct += self.batch_correct
        self.batch_time = self.batch_end_time - self.batch_start_time
        self.batch_start_time = time.time()

    def record_types(self, correct, types):
        for i in range(self.qt_size):
            idx = (types == i).float()
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
        self.writer.add_scalar(f'{self.mode}-1.Total loss', self.epoch_loss / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-2.Total accuracy', self.epoch_correct / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-3.Total time', self.epoch_time, self.epoch_idx)
        self.logs[f"{self.mode}"].append(self.epoch_correct / self.dataset_size)
        for question_type_idx, question_type_name in self.idx_to_question_type.items():
            self.per_question_type['correct'][question_type_name] += self.per_question['correct'][question_type_idx]
            self.per_question_type['number'][question_type_name] += self.per_question['number'][question_type_idx]
        for question_type_name in self.per_question_type['correct'].keys():
            type_accuracy = self.per_question_type['correct'][question_type_name] / self.per_question_type['number'][question_type_name]
            self.per_question_log[question_type_name] = type_accuracy
            self.writer.add_scalar("{}-7. Question '{}' accuracy".format(self.mode, question_type_name), type_accuracy, self.epoch_idx)
        self.update_csv(self.csv_file)

    def log_data(self, image, question, answer, types):
        n = min(image.size()[0], 8)
        question_text = [' '.join([self.idx_to_word[i] for i in q]) for q in question.cpu().numpy()[:n]]
        answer_text = [self.answer_idx_to_word[a] for a in answer.cpu().numpy()[:n]]
        question_type_text = [self.idx_to_question_type[qt] for qt in types.cpu().numpy()[:n]]
        for j, (question, answer, q_type) in enumerate(zip(question_text, answer_text, question_type_text)):
            self.writer.add_text(f'QA{j}', f'Quesetion: {question} / Answer: {answer} / Type: {q_type}', self.epoch_idx)
        self.writer.add_image('Image', make_grid(image[:n], 4), self.epoch_idx)

    def get_epoch_loss(self):
        return self.epoch_loss / self.dataset_size

    def make_header(self):
        record_dict = vars(self.args).copy()
        modes = ["Test", "Train"]
        kinds = ["Latest", "Max", f"MaxRA({self.rolling_average})"]
        values = ["value", "epoch"]
        # values = ["TotalLoss", "TotalAccuracy"]
        for mode in modes:
            for kind in kinds:
                for val in values:
                    record_dict[f"{mode}_{kind}_TotalAcc({val})"] = 0
                record_dict[f"{mode}_{kind}_TotalLoss"] = 0
                for _, question_type in self.idx_to_question_type.items():
                    record_dict[f"{mode}_{kind}_QT_{question_type}"] = 0
        record_dict["Finished"] = False
        for key in self.exclude:
            del record_dict[key]
        return record_dict

    def make_csv(self):
        with open(self.csv_file, 'w') as f:
            w = csv.DictWriter(f, self.header)
            w.writeheader()
        print(f"{self.csv_file} made.")

    def is_record_exist(self):
        with open(self.csv_file, 'r') as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                if row['timestamp'] == self.timestamp:
                    return True
        return False

    def make_record(self):
        self.add_column()
        with open(self.csv_file, 'a', newline='') as f:
            w = csv.DictWriter(f, self.header)
            w.writerow(self.header)
        print(f"Record {self.timestamp} made.")

    def add_column(self):
        tf = NamedTemporaryFile(mode='w', delete=False)
        with open(self.csv_file, 'r') as rf, tf:
            reader = csv.DictReader(rf)
            writer = csv.DictWriter(tf, fieldnames=self.header)
            writer.writeheader()
            for row in reader:
                new_row = OrderedDict()
                for column in self.header:
                    if column in row:
                        new_row[column] = row[column]
                    else:
                        new_row[column] = None
                writer.writerow(new_row)
        shutil.move(tf.name, self.csv_file)

    def update_csv(self, file):
        tf = NamedTemporaryFile(mode='w', delete=False)
        with open(file, 'r') as rf, tf:
            reader = csv.DictReader(rf)
            writer = csv.DictWriter(tf, fieldnames=self.header)
            writer.writeheader()
            for row in reader:
                if row['timestamp'] == self.timestamp:
                    for key, value in self.logs.items():
                        kind = "Latest"
                        row[f"{key}_{kind}_TotalAcc(value)"] = value[-1]
                        row[f"{key}_{kind}_TotalAcc(epoch)"] = self.epoch_idx
                        row[f"{key}_{kind}_TotalLoss"] = self.epoch_loss
                        for question_type, acc in self.per_question_log.items():
                            row[f"{key}_{kind}_QT_{question_type}"] = acc
                        if len(value) == self.rolling_average:
                            kind = "Max"
                            if value[-1] > float(row[f"{key}_{kind}_TotalAcc(value)"]):
                                row[f"{key}_{kind}_TotalAcc(value)"] = value[-1]
                                row[f"{key}_{kind}_TotalAcc(epoch)"] = self.epoch_idx
                                row[f"{key}_{kind}_TotalLoss"] = self.epoch_loss
                                for question_type, acc in self.per_question_log.items():
                                    row[f"{key}_{kind}_QT_{question_type}"] = acc
                            kind = f"MaxRA({self.rolling_average})"
                            if mean(value) > float(row[f"{key}_{kind}_TotalAcc(value)"]):
                                row[f"{key}_{kind}_TotalAcc(value)"] = mean(value)
                                row[f"{key}_{kind}_TotalAcc(epoch)"] = self.epoch_idx
                                row[f"{key}_{kind}_TotalLoss"] = self.epoch_loss
                                for question_type, acc in self.per_question_log.items():
                                    row[f"{key}_{kind}_QT_{question_type}"] = acc
                writer.writerow(row)
        shutil.move(tf.name, file)

    def finish(self):
        tf = NamedTemporaryFile(mode='w', delete=False)
        with open(self.csv_file, 'r') as rf, tf:
            reader = csv.DictReader(rf)
            writer = csv.DictWriter(tf, fieldnames=reader.fieldnames)
            for row in reader:
                if row['timestamp'] == self.timestamp:
                    row["Finished"] = True
                writer.writerow(row)
        shutil.move(tf.name, self.csv_file)

def mean(list):
    return sum(list)/float(len(list))
