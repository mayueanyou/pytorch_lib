import torch,random,copy,os
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from path import*
from trainer import Trainer, Net
from net import*

random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)


def main():
    net = Net(net = FNN_1(),load = False)

    training_data = datasets.MNIST(root="../../datasets",train=True,download=True,transform=ToTensor(),)
    training_data, validate_data = torch.utils.data.random_split(training_data, [50000, 10000])
    test_data = datasets.MNIST(root="../../datasets",train=False,download=True,transform=ToTensor(),)
    trainer = Trainer(training_data,test_data,validate_data,net)
    trainer.update_extra_info()
    trainer.train_test(100)

if __name__ == '__main__':
    main()