import torch,random,copy,os
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

def cnn_cell(input_channel,output_channel,kernel_size,stride,pedding,relu=True,bias=True):
    if relu:
        return nn.Sequential(
        nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias),
        nn.BatchNorm2d(output_channel),nn.ReLU())
    else:
        return nn.Sequential(
        nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias),
        nn.BatchNorm2d(output_channel),)

class FNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'FNN_1'
        self.input_size = (1,28,28)

        self.fc1 = nn.Linear(784, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x,x


class CNN_1(nn.Module):
    class Cell(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn1 = cnn_cell(1,10,3,1,1)
            self.cnn2 = cnn_cell(10,10,3,1,1)
            self.cnn3 = cnn_cell(10,5,3,1,1)
            self.cnn4 = cnn_cell(5,5,3,1,1)
            self.cnn5 = cnn_cell(5,5,3,1,1)

            self.fc1 = nn.Linear(1620,1)

        def forward(self,x):
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x)
            x = self.cnn4(x)
            x = self.cnn5(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x

    def __init__(self):
        super().__init__()
        self.name = 'CNN_1_t'
        self.input_size = (1,28,28)
        self.input_dim = self.input_size[0]*self.input_size[1]*self.input_size[2]

        self.layers = nn.ModuleList([self.Cell() for i in range(10)])

    def forward(self,x):
        result = []
        for layer in self.layers: result.append(layer(x))
        x = torch.cat(result, dim=1)
        return x,x
    
class ResNet_1(nn.Module):
    class Cell(nn.Module):
        def __init__(self, input_channel, output_channel, stride, downsample=None):
            super().__init__()
            self.conv1 = cnn_cell(input_channel,output_channel,3,stride,1)
            self.conv2 = cnn_cell(output_channel,output_channel,3,1,1,relu=False)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample is not None: residual = self.downsample(residual)
            out += residual
            out = F.relu(out)
            return out
        
    def __init__(self):
        super().__init__()
        self.name = 'ResNet_1'
        self.input_size = (1,28,28)

        self.conv1 = cnn_cell(self.input_size[0],64,7,2,3,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 64, 2, 1)
        self.layer2 = self.make_layer(64, 128, 2, 2)
        self.layer3 = self.make_layer(128, 256, 2, 2)
        self.layer4 = self.make_layer(256, 512, 2, 2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 10)

    def make_layer(self,input_channel,output_channel,number,stride):
        downsample = None
        if stride != 1 or input_channel != output_channel:
            downsample = cnn_cell(input_channel,output_channel,1,stride,0,relu=False)
        layers = []
        layers.append(self.Cell(input_channel, output_channel, stride, downsample))
        for i in range(1, number):
            layers.append(self.Cell(output_channel, output_channel,1))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here: disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

class Transformer_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Transformer_1'
        self.input_size = (1,28,28)

        self.fc1 = nn.Linear(784, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x,x  