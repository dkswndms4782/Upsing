import torch.nn as nn
def get_criterion(pred, target):
    loss = nn.CrossEntropyLoss()
    return loss(pred, target)