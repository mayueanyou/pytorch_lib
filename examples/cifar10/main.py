import os,sys,torch,random,argparse
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Compose

from module import*
import pytorch_lib as pl

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)


dataset_path = "~/datasets"
current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def main(name):
    #tf = Compose([ToTensor(),Grayscale()])
    tf = ToTensor()
    dataset = pl.CIFAR10(dataset_path,batch_size=128,training_transform=tf,test_transform=tf)
    training_data,test_data,validate_data = dataset.loaders()
    net = pl.Net(net = getattr(sys.modules[__name__], name)(),load = False,model_folder_path=current_path+'/model/',loss=pl.CELoss())
    trainer = pl.Trainer(net,training_data,test_data,validate_data)
    trainer.train_test(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="None")
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args.net)