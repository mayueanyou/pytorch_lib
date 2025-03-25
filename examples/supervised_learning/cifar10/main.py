import os,sys,torch,random,argparse
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Compose

from module import*
import pytorch_lib as ptl

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)


dataset_path = "/home/yma183/datasets/CIFAR10"
current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def main(args):
    #tf = Compose([ToTensor(),Grayscale()])
    tf = ToTensor()
    dataset = ptl.CIFAR10(dataset_path,training_transform=tf,test_transform=tf)
    training_data,test_data,validate_data = dataset.loaders(batch_size=128)
    net = ptl.Net(net = getattr(sys.modules[__name__], args.net)(),load = False,model_folder_path=current_path+'/model/',loss=ptl.CELoss())
    trainer = ptl.Trainer(net,training_data,test_data,validate_data)
    trainer.train_test(100)

def save_image(args):
    tf = ToTensor()
    dataset = ptl.CIFAR10(dataset_path,training_transform=tf,test_transform=tf)
    dataset.save_images()

def test(args):
    tf = ToTensor()
    dataset = ptl.CIFAR10(dataset_path,training_transform=tf,test_transform=tf)
    new_dataset = ptl.dataset_lib.select_by_label(dataset.test_dataset,[1,2])
    print(dataset.test_dataset.data.shape)
    print(new_dataset.data.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="None")
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args)