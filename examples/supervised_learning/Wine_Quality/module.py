import os,torch
from torch import nn
import torch.nn.functional as F

from pytorch_template import*

class FNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        self.input_size = (11)

        self.fc1 = nn.Linear(11, 1)

    def forward(self,x):
        x = self.fc1(x)
        return x,x