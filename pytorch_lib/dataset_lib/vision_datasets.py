import os,sys,json,pathlib,shutil
import pandas as pd
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Compose,Resize
from abc import ABC, abstractmethod
from tqdm import tqdm

from .datasetloader import *

# class VisionDataset(ABC):
#     def __init__(self) -> None:
#         super().__init__()
#         self.name = type(self).__name__
    
#     def get_classes_from_file(self):
#         current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
#         with open(current_path + f'/supplement/{self.name}/classes.txt', 'r') as file: self.classes_original = json.loads(file.read())
#         self.classes = list(self.classes_original.values())
    
#     @abstractmethod
#     def load(self):pass
    
#     @abstractmethod
#     def datas(self):pass
    
#     @abstractmethod
#     def loaders(self):pass
    
#     def classes(self):
#         return self.classes


class ImageFolder:
    def __init__(self,dataset_path:str,transform=ToTensor()) -> None:
        self.root = dataset_path
        self.data = datasets.ImageFolder(root = self.root ,transform = transform)
    
    def loaders(self,batch_size=64):
        dataloader = DataLoader(self.data, batch_size = batch_size)

        return dataloader

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
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),load_data=True) -> None:
        super().__init__()
        if load_data: self.load(dataset_path,training_transform=training_transform,test_transform=test_transform)
        print('Dataset: ',self.name)
        self.get_classes_from_file()
    
    def load(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor()):
        self.train_dataset = datasets.CIFAR100(root=dataset_path,train=True,download=True,transform=training_transform)
        self.test_dataset = datasets.CIFAR100(root=dataset_path,train=False,download=True,transform=test_transform)
        self.dataset_loader = DatasetLoader(self.train_dataset,self.test_dataset)

    def datas(self,target_list=None,label_setup=None):
        return self.dataset_loader.get_datas(target_list=target_list,label_setup=label_setup)
    
    def loaders(self,target_list=None,label_setup=None,batch_size=64):
        return self.dataset_loader.get_loaders(target_list=target_list,label_setup=label_setup,batch_size=batch_size)
    
class ImageNet2012:
    def __init__(self,dataset_path,training_transform=Compose([Resize((224,224)),ToTensor()]),test_transform=Compose([Resize((224,224)),ToTensor()]),target_list=None,label_setup=None) -> None:
        self.training_data = datasets.ImageNet(root=dataset_path,split='train',transform=training_transform)
        self.test_data = datasets.ImageNet(root=dataset_path,split='val',transform=test_transform)
        self.validate_data = datasets.ImageNet(root=dataset_path,split='val',transform=test_transform)
        self.name = 'ImageNet2012'
        print('Dataset: ',self.name)
        self.get_classes()
    
    def get_classes(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + f'/supplement/{self.name}/classes.txt', 'r') as file: self.classes_original = json.loads(file.read())
        self.classes = list(self.classes_original.values())
    
    def loaders(self,batch_size=64):
        train_dataloader = DataLoader(self.training_data, batch_size = batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size = batch_size)
        validate_dataloader = DataLoader(self.validate_data, batch_size = batch_size)
        return train_dataloader,test_dataloader,validate_dataloader

class CUB200():
    def __init__(self,dataset_path,training_transform=Compose([Resize((224,224)),ToTensor()]),test_transform=Compose([Resize((224,224)),ToTensor()])) -> None:
        super().__init__()
        self.root = dataset_path
        self.name = 'CUB200'
        print('Dataset: ',self.name)
        self.training_data = datasets.ImageFolder(root = self.root + f'/train',transform = training_transform)
        self.test_data = datasets.ImageFolder(root = self.root + f'/test',transform = test_transform)
        #self.get_classes_from_file()
        self.load_metadata()
    
    def loaders(self,batch_size=64):
        train_dataloader = DataLoader(self.training_data, batch_size = batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size = batch_size)
        return train_dataloader,test_dataloader
    
    def load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),sep=' ', names=['img_id', 'is_training_img'])
        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.train_data_pd = self.data[self.data.is_training_img == 1]
        self.test_data_pd = self.data[self.data.is_training_img == 0]
        
    def split_train_test(self):
        if not os.path.exists(self.root + '/train'): os.makedirs(self.root + '/train')
        if not os.path.exists(self.root + '/test'): os.makedirs(self.root + '/test')
        
        def move_file(dataset_pd,folder):
            for id in tqdm(range(200)):
                if not os.path.exists(self.root + f'/{folder}/cub_{id}'): os.makedirs(self.root + f'/{folder}/cub_{id}')
                
                cls_data_pd = dataset_pd[dataset_pd.target == id+1]
                for i in range(len(cls_data_pd)):
                    data_pd = cls_data_pd.iloc[i]
                    shutil.copy(self.root + f'/CUB_200_2011/images/{data_pd.filepath}',self.root + f'/{folder}/cub_{id}' )
        
        move_file(self.train_data_pd,'train')
        move_file(self.test_data_pd,'test')

class ImageNetRC:
    def __init__(self,dataset_path,training_transform=Compose([Resize((224,224)),ToTensor()]),test_transform=Compose([Resize((224,224)),ToTensor()])) -> None:
        self.root = dataset_path
        self.training_data = datasets.ImageFolder(root = self.root + f'/train',transform = training_transform)
        self.test_data = datasets.ImageFolder(root = self.root + f'/test',transform = test_transform)
        self.name = 'ImageNetRC'
        print('Dataset: ',self.name)
        self.get_classes()
    
    def get_classes(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + f'/supplement/ImageNet-R/classes.txt', 'r') as file: self.classes_original = json.loads(file.read())
        self.classes = list(self.classes_original.values())
    
    def loaders(self,batch_size=64):
        train_dataloader = DataLoader(self.training_data, batch_size = batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size = batch_size)
        return train_dataloader,test_dataloader