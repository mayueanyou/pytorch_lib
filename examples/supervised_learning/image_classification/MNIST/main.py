import os,sys,torch,random,argparse
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lib as ptl
from module import*

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

dataset_path = "/datasets/MNIST"
current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def main():
    net = ptl.Net(net = FNN_1(),
                  load = False,
                  model_folder_path = current_path + '/model/',
                  loss=ptl.CELoss())
    
    dataset = ptl.MNIST(dataset_path)
    train_dl,test_dl,validate_dl = dataset.get_loaders()
    trainer = ptl.Trainer(net,train_dl,test_dl,validate_dl)
    trainer.train_test(10)

if __name__ == '__main__':
    main()