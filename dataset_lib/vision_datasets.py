import os,sys,json
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Compose,Resize
from abc import ABC, abstractmethod

from .datasetloader import DatasetLoader

class VisionDataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
    
    def get_classes_from_file(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + f'/classes/{self.name}/classes.txt', 'r') as file: self.classes_original = json.loads(file.read())
        self.classes = list(self.classes_original.values())
    
    @abstractmethod
    def load(self):pass
    
    @abstractmethod
    def datas(self):pass
    
    @abstractmethod
    def loaders(self):pass
    
    def classes(self):
        return self.classes

class MNIST:
    def __init__(self,dataset_path:str,training_transform=ToTensor(),test_transform=ToTensor(),target_list=None,label_setup=None,batch_size=64) -> None:
        self.dataset_loader = DatasetLoader(datasets.MNIST(root=dataset_path,train=True,download=True,transform=training_transform),
                                            datasets.MNIST(root=dataset_path,train=False,download=True,transform=test_transform))
        self.target_list = target_list
        self.label_setup = label_setup
        self.batch_size = batch_size
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.name = 'MNIST'
        print('Dataset: ',self.name)
    
    def datas(self):
        return self.dataset_loader.get_datas(target_list=self.target_list,label_setup=self.label_setup)
    
    def loaders(self):
        return self.dataset_loader.get_loaders(target_list=self.target_list,label_setup=self.label_setup,batch_size=self.batch_size)

class CIFAR10:
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),target_list=None,label_setup=None,batch_size=64) -> None:
        self.dataset_loader = DatasetLoader(datasets.CIFAR10(root=dataset_path,train=True,download=True,transform=training_transform),
                                            datasets.CIFAR10(root=dataset_path,train=False,download=True,transform=test_transform))
        self.target_list = target_list
        self.label_setup = label_setup
        self.batch_size = batch_size
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.name = 'CIFAR10'
        print('Dataset: ',self.name)
    
    def datas(self):
        return self.dataset_loader.get_datas(target_list=self.target_list,label_setup=self.label_setup)
    
    def loaders(self):
        return self.dataset_loader.get_loaders(target_list=self.target_list,label_setup=self.label_setup,batch_size=self.batch_size)

class CIFAR100(VisionDataset):
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),target_list=None,label_setup=None,batch_size=64,load_data=True) -> None:
        super().__init__()
        if load_data: self.load(dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),target_list=None,label_setup=None,batch_size=64)
        print('Dataset: ',self.name)
        self.get_classes_from_file()
    
    def load(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),target_list=None,label_setup=None,batch_size=64):
        self.dataset_loader = DatasetLoader(datasets.CIFAR100(root=dataset_path,train=True,download=True,transform=training_transform),
                                            datasets.CIFAR100(root=dataset_path,train=False,download=True,transform=test_transform))
        self.target_list = target_list
        self.label_setup = label_setup
        self.batch_size = batch_size

    def datas(self):
        return self.dataset_loader.get_datas(target_list=self.target_list,label_setup=self.label_setup)
    
    def loaders(self):
        return self.dataset_loader.get_loaders(target_list=self.target_list,label_setup=self.label_setup,batch_size=self.batch_size)
    
class ImageNet2012:
    def __init__(self,dataset_path,training_transform=Compose([Resize((224,224)),ToTensor()]),test_transform=Compose([Resize((224,224)),ToTensor()]),target_list=None,label_setup=None,batch_size=64) -> None:
        self.training_data = datasets.ImageNet(root=dataset_path,split='train',transform=training_transform)
        self.test_data = datasets.ImageNet(root=dataset_path,split='val',transform=test_transform)
        self.validate_data = datasets.ImageNet(root=dataset_path,split='val',transform=test_transform)
        self.batch_size = batch_size
        self.name = 'ImageNet2012'
        print('Dataset: ',self.name)
        self.get_classes()
    
    def get_classes(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + f'/classes/{self.name}/classes.txt', 'r') as file: self.classes_original = json.loads(file.read())
        self.classes = list(self.classes_original.values())
    
    def loaders(self):
        train_dataloader = DataLoader(self.training_data, batch_size = self.batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size = self.batch_size)
        validate_dataloader = DataLoader(self.validate_data, batch_size = self.batch_size)
        return train_dataloader,test_dataloader,validate_dataloader