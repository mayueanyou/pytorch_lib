import os,sys,torch,random
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

file_path=os.path.abspath(__file__)
current_path =  os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")
upper_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
upper_upper_path = os.path.abspath(os.path.dirname(upper_path) + os.path.sep + ".")
sys.path.append(upper_path)
from trainer import Trainer, Net
from net import*

random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

def check_gpu():
    if torch.cuda.is_available():print(torch.cuda.get_device_name(0))
    else:print('No GPU')


def main():
    #net = Net(net = FNN_1(),load = False,model_path=current_path)
    net = Net(net = ResNet_1(),load = False,model_path=current_path)

    training_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=True,download=True,transform=ToTensor(),)
    training_data, validate_data = torch.utils.data.random_split(training_data, [50000, 10000])
    test_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=False,download=True,transform=ToTensor(),)
    trainer = Trainer(training_data,test_data,validate_data,net)
    trainer.update_extra_info()
    trainer.train_test(10)

if __name__ == '__main__':
    check_gpu()
    main()