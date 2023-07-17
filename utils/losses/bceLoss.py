import torch
import torch.nn as nn


def BCE_loss(pred,label):
    bce_loss = nn.BCELoss(reduction="mean")
    bce_out = bce_loss(pred, label)
    return bce_out
