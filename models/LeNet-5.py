import torch
import torch.nn as nn
import torch.nn.Functional as F

h, w = 32, 32


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2D(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(120, 84, bias=False)
        self.fc2 = nn.Linear(84, 10, bias=False)

    def forward(x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
