import os,sys,torch,random,argparse
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
from trainer import*
from net import*
from module import*
from datasetloader import*
from loss import*

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

def prepare_loaders(transform,target_list=None,label_setup=None,batch_size=64):
    training_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=True,download=True,transform=transform)
    test_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=False,download=True,transform=transform)
    dataset_loader = DatasetLoader(training_data,test_data)
    training_data,test_data,validate_data = dataset_loader.get_loaders(target_list=target_list,label_setup=label_setup,batch_size=batch_size)
    return training_data,test_data,validate_data

def train(net,epoch,transform,target_list=None,label_setup=None):
    training_data,test_data,validate_data = prepare_loaders(transform,target_list=target_list,label_setup=label_setup)
    trainer = Trainer(net,training_data,test_data,validate_data)
    trainer.train_test(epoch)

def test(net,transform,target_list=None,label_setup=None):
    training_data,test_data,validate_data = prepare_loaders(transform,target_list=target_list,label_setup=label_setup)
    trainer = Trainer(net,training_data,test_data,validate_data)
    trainer.test()
    
def main(name):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    net = Net(net = getattr(sys.modules[__name__], name)(),load = True,model_folder_path=current_path+'/model/',loss=CELoss())
    target_list=None
    label_setup=None

    #train(net,10,ToTensor(),target_list=target_list,label_setup=label_setup)
    #test(net,ToTensor(),target_list=target_list,label_setup=label_setup)
    
    training_data,test_data,validate_data = prepare_loaders(ToTensor(),target_list=target_list,label_setup=label_setup,batch_size=-1)
    net.get_confusion_matrix(test_data,classes)
    

def test_new():
    def test_net(num,data,net):
        with torch.no_grad():
            print(net.net.name)
            pred,feature = net.net(data)
            pred = F.softmax(pred,dim=1)
            #print(pred)
            sum = torch.sum(pred, 0)
            max_idxs = pred.argmax(1)
            logits = torch.zeros((len(pred),10))
            for i in range(len(pred)):
                logits[i][int(max_idxs[i])] = 1
            logits=torch.sum(logits, 0)
            print(sum)
            print(logits)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net1 = Net(net = FNN_1(),load = True,model_path=current_path+'/model/')
    net2 = Net(net = ResNet_1(),load = True,model_path=current_path+'/model/')
    #net2 = Net(net = FNN_2(),load = True,model_path=current_path+'/model/')
    net1.net.eval()
    net2.net.eval()
    #training_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=True,download=True,transform=ToTensor(),)
    test_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=False,download=True,transform=ToTensor(),)
    #test_data = reset_dataset(test_data,[0])
    for i in range(10):
        num = i
        idx = test_data.targets==num
        data = test_data.data[idx]
        data = data[:,None,:]
        data = data.to(torch.float32)
        data = data.to(device)
        data = data/255
        #print(tmp)
        print(num,len(data))
        #test_net(num,data,net1)
        test_net(num,data,net2)
        print()
        #break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str)
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args.net)