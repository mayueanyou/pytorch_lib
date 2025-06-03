import os,sys,torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,pedding=1,activation_function=True,bias=True):
        super().__init__()
        layers = [nn.Conv2d(in_channel,out_channel,kernel_size,stride,pedding,bias=bias),nn.BatchNorm2d(out_channel,track_running_stats=True)]
        if activation_function: layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layers(x)

def cnn_cell(input_channel,output_channel,kernel_size,stride,pedding,activation_function=True,bias=True):
    layers = [nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias),nn.BatchNorm2d(output_channel,track_running_stats=True)]
    #if activation_function: layers.append(nn.ReLU())
    if activation_function: layers.append(nn.GELU())
    return nn.Sequential(*layers)