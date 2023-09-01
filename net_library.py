import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, layers,transform=None):
        super().__init__()
        self.layers = layers
        self.transform = transform

    def forward(self, x):
        residual = x
        if self.transform is not None: residual = self.transform(residual)
        return self.layers(x) + residual

def cnn_cell(input_channel,output_channel,kernel_size,stride,pedding,relu=True,bias=True):
    layers = [nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias),
              nn.BatchNorm2d(output_channel,track_running_stats=True)]
    if relu: layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def initialize_cnn(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**.5)