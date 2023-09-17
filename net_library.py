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

def residual_cell(input_channel,output_channel,number,stride):
        def create_cell(input_channel,output_channel,stride):
            layers = [cnn_cell(input_channel,output_channel,3,stride,1),
                      cnn_cell(output_channel,output_channel,3,1,1,relu=False)]
            return nn.Sequential(*layers)
        
        downsample = None
        if stride != 1 or input_channel != output_channel:
            downsample = cnn_cell(input_channel,output_channel,1,stride,0,relu=False)
        layers = []
        layers.append(Residual(create_cell(input_channel,output_channel,stride),downsample))
        for i in range(1, number):
            layers.append(Residual(create_cell(output_channel,output_channel,1)))
        return nn.Sequential(*layers)

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