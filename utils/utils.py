import torch
import torch.nn.functional as F

def baseline_encode(images, questions):
    device = images.get_device()
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    questions = questions.unsqueeze(2).unsqueeze(3).expand(n, hd, h, w)
    images = torch.cat([images, x_coordinate, y_coordinate, questions], 1).view(n, -1, o).transpose(1, 2)
    return images


def rn_encode(images, questions):
    device = images.get_device()
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-1, 1, h).view(1, h, 1, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    images = images.view(n, c, o).transpose(1, 2)
    images = torch.cat([images, x_coordinate, y_coordinate], 2)
    images1 = images.unsqueeze(1).expand(n, o, o, c + 2).contiguous()
    images2 = images.unsqueeze(2).expand(n, o, o, c + 2).contiguous()
    questions = questions.unsqueeze(1).unsqueeze(2).expand(n, o, o, hd)
    # pairs = torch.cat([images1, images2, questions], 3).view(n, o**2, -1)
    pairs = torch.cat([images1, images2, questions], 3)
    return pairs


def lower_sum(relations):
    device = relations.get_device()
    n, h, w, l = relations.size()
    mask = torch.ones([h, w]).tril().unsqueeze(0).unsqueeze(3).to(device)
    relations = relations * mask
    return relations.sum(2)


def sarn_encode(objects, code):
    device = objects.get_device()
    n, c, h, w = objects.size()
    o = h * w
    hd = code.size(1)
    x_coordinate = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    coordinate_encoded = torch.cat([objects, x_coordinate, y_coordinate], 1)
    question = code.view(n, hd, 1, 1).expand(n, hd, h, w)
    question_encoded = torch.cat([coordinate_encoded, question], 1).view(n, -1, o).transpose(1, 2)
    return coordinate_encoded.view(n, -1, o).transpose(1, 2), question_encoded


def sarn_pair(coordinate_encoded, question_encoded, logits):
    selection = F.softmax(logits.squeeze(2), dim=1)
    selected = torch.bmm(selection.unsqueeze(1), coordinate_encoded).expand_as(coordinate_encoded)
    pairs = torch.cat([question_encoded, selected], 2)
    return pairs


