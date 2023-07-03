import torch
from config import config
from torch import nn as nn
from torchvision import models


class BaseNetwork(nn.Module):

    def __init__(self, config: config) -> None:
        super().__init__()
        network = models.vgg.vgg16(pretrained=True)
        self.feature = network.features
        self.pool = network.avgpool
        self.dnn = network.classifier
        self.adaptor = nn.Linear(1000, 512)
        self.classifier = nn.Linear(512, config.class_num)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dnn(x)
        x = torch.relu(x)
        x = self.adaptor(x)
        x = torch.relu(x)
        x = self.classifier(x)
        return x
