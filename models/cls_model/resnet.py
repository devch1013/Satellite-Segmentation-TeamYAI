from torchvision import models
import torch
import torch.nn as nn


class ResnetCls(nn.Module):
    def __init__(self):
        super(ResnetCls, self).__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        num_classes = 1
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze()
