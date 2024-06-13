import os,sys,torch
from torch import nn
import torch.nn.functional as F
from pytorch_template import*
from einops import rearrange

input_size = (32,32,3)

class ResNet_original(nn.Module):
    def __init__(self,in_channel=3,class_number=10):
        super().__init__()
        self.name = type(self).__name__
        self.input_size = input_size

        self.conv1 = cnn_cell(in_channel,64,3,1,1,bias=False)
        self.layer1 = residual_cell(64, 64, 2, 1)
        self.layer2 = residual_cell(64, 128, 2, 2)
        self.layer3 = residual_cell(128, 256, 2, 2)
        self.layer4 = residual_cell(256, 512, 2, 2)
        self.fc = nn.Linear(512, class_number)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits#, probas