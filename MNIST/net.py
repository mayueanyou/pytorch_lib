import torch,random,copy,os
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

def cnn_cell(input_channel,output_channel,kernel_size,pedding):
    return nn.Sequential(
        nn.Conv2d(input_channel,output_channel,kernel_size,pedding),
        nn.BatchNorm2d(output_channel),
        nn.ReLU())

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


class CNN_1_t(nn.Module):
    class Cell(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn1 = cnn_cell(1,10,3,1)
            self.cnn2 = cnn_cell(10,10,3,1)
            self.cnn3 = cnn_cell(10,5,3,1)
            self.cnn4 = cnn_cell(5,5,3,1)
            self.cnn5 = cnn_cell(5,5,3,1)

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