import os,sys,torch,random,argparse
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lib as pl
from module import*

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

dataset_path = "~/datasets"
current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def main(name):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    #net = pt.Net(net = getattr(sys.modules[__name__], name)(),load = False,model_folder_path=current_path+'/model/',loss=CELoss(),postfix='extra_token')
    net = pl.Net(net = getattr(sys.modules[__name__], name)(),load = False,model_folder_path=current_path+'/model/',loss=CELoss())
    target_list=None
    label_setup=None
    dataset = pl.MNIST(dataset_path)
    training_data,test_data,validate_data = dataset.loaders()
    trainer = pl.Trainer(net,training_data,test_data,validate_data)
    trainer.train_test(10)
    #training_data,test_data,validate_data = prepare_loaders(ToTensor(),target_list=target_list,label_setup=label_setup,batch_size=-1)
    #net.get_confusion_matrix(test_data,classes,current_path+'/image')

def plot_test(args):
    data = torch.rand(2,1,28,28)
    pl.plot_patches(data,28,7)

def condor(name):
    import condor_src
    condor_src.condor_submit(current_path + '/condor', __file__, '-f main -net FNN_1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="None")
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args.net)