import os,sys,torch,random,argparse
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo 

file_path=os.path.abspath(__file__)
current_path =  os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")
upper_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
upper_upper_path = os.path.abspath(os.path.dirname(upper_path) + os.path.sep + ".")
sys.path.append(upper_upper_path)
from pytorch_template import*

from module import*

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

def prepare_loaders(target_list=None,label_setup=None,batch_size=64):
    wine_quality = fetch_ucirepo(id=186) 
    X = torch.tensor(wine_quality.data.features.values,dtype=torch.float)
    y = torch.tensor(wine_quality.data.targets.values,dtype=torch.float)
    dataset = CustomDataset(X,y)
    
    training_data = dataset
    test_data = dataset
    dataset_loader = DatasetLoader(training_data,test_data)
    training_data,test_data,validate_data = dataset_loader.get_loaders(target_list=target_list,label_setup=label_setup,batch_size=batch_size)
    return training_data,test_data,validate_data

def train(net,epoch,target_list=None,label_setup=None):
    training_data,test_data,validate_data = prepare_loaders(target_list=target_list,label_setup=label_setup)
    trainer = Trainer(net,training_data,test_data,validate_data)
    trainer.train_test(epoch)

def test(net,target_list=None,label_setup=None):
    training_data,test_data,validate_data = prepare_loaders(target_list=target_list,label_setup=label_setup)
    trainer = Trainer(net,training_data,test_data,validate_data)
    trainer.test()
    
def main(args):
    net = Net(net = getattr(sys.modules[__name__], args.net)(),load = False,model_folder_path=current_path+'/model/',loss=MSELoss())

    train(net,10)
    test(net)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str)
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args)