import os,sys,torch
from torch import nn
import torch.nn.functional as F
from . import*

class Residual(nn.Module):
    def __init__(self, layers,transform=None,learnable_rate={'x':False,'tx':False}):
        super().__init__()
        self.layers = layers
        self.px = nn.Parameter(torch.ones(1)) if learnable_rate['x'] else 1
        self.tpx = nn.Parameter(torch.ones(1)) if learnable_rate['tx'] else 1
        self.transform = transform

    def forward(self, x):
        residual = x if self.transform is None else self.transform(x)
        return self.layers(x) * self.tpx + residual * self.px

def residual_cell(input_channel,output_channel,number,stride,learnable_rate={'x':False,'tx':False}):
    def create_cell(input_channel,output_channel,stride):
        layers = [cnn_cell(input_channel,output_channel,3,stride,1), cnn_cell(output_channel,output_channel,3,1,1,activation_function=False)]
        return nn.Sequential(*layers)
    
    downsample = None
    if stride != 1 or input_channel != output_channel:
        downsample = cnn_cell(input_channel,output_channel,1,stride,0,activation_function=False)
    layers = []
    layers.append(Residual(create_cell(input_channel,output_channel,stride),downsample,learnable_rate=learnable_rate))
    for i in range(1, number):
        layers.append(Residual(create_cell(output_channel,output_channel,1),learnable_rate=learnable_rate))
    return nn.Sequential(*layers)

class ResNet_18(nn.Module):
    def __init__(self,input_channle,num_cls):
        super().__init__()
        self.name = type(self).__name__

        self.conv1 = cnn_cell(input_channle,64,7,2,3,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = residual_cell(64, 64, 2, 1)
        self.layer2 = residual_cell(64, 128, 2, 2)
        self.layer3 = residual_cell(128, 256, 2, 2)
        self.layer4 = residual_cell(256, 512, 2, 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(128, num_cls)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits