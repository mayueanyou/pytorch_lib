import os,sys,copy,torch,random,cv2,torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor

class MNIST:
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),target_list=None,label_setup=None,batch_size=64) -> None:
        self.dataset_loader = DatasetLoader(datasets.MNIST(root=dataset_path,train=True,download=True,transform=training_transform),
                                            datasets.MNIST(root=dataset_path,train=False,download=True,transform=test_transform))
        self.target_list = target_list
        self.label_setup = label_setup
        self.batch_size = batch_size
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.name = 'MNIST'
    
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
    
    def datas(self):
        return self.dataset_loader.get_datas(target_list=self.target_list,label_setup=self.label_setup)
    
    def loaders(self):
        return self.dataset_loader.get_loaders(target_list=self.target_list,label_setup=self.label_setup,batch_size=self.batch_size)

class CIFAR100:
    def __init__(self,dataset_path,training_transform,test_transform,target_list=None,label_setup=None,batch_size=64) -> None:
        self.dataset_loader = DatasetLoader(datasets.CIFAR100(root=dataset_path,train=True,download=True,transform=training_transform),
                                            datasets.CIFAR100(root=dataset_path,train=False,download=True,transform=test_transform))
        self.target_list = target_list
        self.label_setup = label_setup
        self.batch_size = batch_size
        self.classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.name = 'CIFAR100'
    
    def datas(self):
        return self.dataset_loader.get_datas(target_list=self.target_list,label_setup=self.label_setup)
    
    def loaders(self):
        return self.dataset_loader.get_loaders(target_list=self.target_list,label_setup=self.label_setup,batch_size=self.batch_size)
    
class ImageNet2012:
    def __init__(self,dataset_path,training_transform,test_transform,target_list=None,label_setup=None,batch_size=64) -> None:
        self.training_data = datasets.ImageNet(root=dataset_path,split='train',transform=training_transform)
        self.test_data = datasets.ImageNet(root=dataset_path,split='val',transform=test_transform)
        self.validate_data = datasets.ImageNet(root=dataset_path,split='val',transform=test_transform)
        self.batch_size = batch_size
        self.classes = None
        self.name = 'ImageNet2012'
    
    def loaders(self):
        train_dataloader = DataLoader(self.training_data, batch_size = self.batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size = self.batch_size)
        validate_dataloader = DataLoader(self.validate_data, batch_size = self.batch_size)
        return train_dataloader,test_dataloader,validate_dataloader

class CustomDataset(Dataset):
    def __init__(self, data, targets, normalize = False):
        self.data = data
        self.targets = targets
        if normalize: self.normalize()
        self.data_shape = data.shape
        self.targets_shape = targets.shape
        print(f'data shape: {self.data_shape}')
        print(f'targets shape: {self.targets_shape}')
    
    def normalize(self):
        self.data = torch.div(self.data, torch.max(self.data,dim=0).values)
        self.targets = torch.div(self.targets, torch.max(self.targets,dim=0).values)    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class CustomDatasetLoader:
    def __init__(self,train_data,test_data,validate_data,batch_size=64) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.validate_data = validate_data
        self.batch_size = batch_size
    
    def loaders(self):
        train_dataloader = DataLoader(self.train_data, batch_size = self.batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size = self.batch_size)
        validate_dataloader = DataLoader(self.validate_data, batch_size = self.batch_size)
        return train_dataloader,test_dataloader,validate_dataloader

class DatasetLoader():
    def __init__(self,train_data,test_data) -> None:
        self.validate_rate = 0.2
        self.train_data = train_data
        if not torch.is_tensor(self.train_data.targets): self.train_data.targets = torch.tensor(self.train_data.targets)
        self.test_data = test_data
        if not torch.is_tensor(self.test_data.targets): self.test_data.targets = torch.tensor(self.test_data.targets)
    
    def print_info(self,train,test,validate,batch_size):
        print(f'batch size: {batch_size}')
        batch_size = abs(batch_size)
        print(f'data in total:  train[{len(train)}] test[{len(test)}] validate[{len(validate)}]')
        print(f'batchs in total: train[{len(train)//batch_size}] test[{len(test)//batch_size}] validate[{len(validate)//batch_size}]\n')
    
    def get_datas(self,target_list=None,label_setup=None):
        training_data,test_data,validate_data = self.dataset_reset(target_list,label_setup)
        return training_data,test_data,validate_data
    
    def get_loaders(self,target_list=None,label_setup=None,batch_size = 64):
        training_data,test_data,validate_data = self.dataset_reset(target_list,label_setup)
        self.print_info(training_data,test_data,validate_data,batch_size)
        if batch_size == -1:
            train_dataloader = DataLoader(training_data, batch_size = len(training_data))
            test_dataloader = DataLoader(test_data, batch_size = len(test_data))
            validate_dataloader = DataLoader(validate_data, batch_size = len(validate_data))
        else:
            train_dataloader = DataLoader(training_data, batch_size = batch_size)
            test_dataloader = DataLoader(test_data, batch_size = batch_size)
            validate_dataloader = DataLoader(validate_data, batch_size = batch_size)
        return train_dataloader,test_dataloader,validate_dataloader
        
    def dataset_select_bylabel(self,dataset,targets):
        idx = sum(dataset.targets==i for i in targets).bool()
        dataset_new = copy.deepcopy(dataset)
        dataset_new.data = dataset_new.data[idx]
        dataset_new.targets = dataset_new.targets[idx]
        return dataset_new

    def dataset_change_label(self,dataset,label_setup):
        dataset_new = copy.deepcopy(dataset)
        for setup in label_setup:
            idx = sum(dataset.targets==i for i in setup[0]).bool()
            dataset_new.targets[idx] = setup[1]
        return dataset_new
    
    def dataset_separate_validate(self,dataset):
        split_number = int(len(dataset)*self.validate_rate)
        train = copy.deepcopy(dataset)
        train.data = train.data[split_number:]
        train.targets = train.targets[split_number:]
        validate = copy.deepcopy(dataset)
        validate.data = validate.data[:split_number]
        validate.targets = validate.targets[:split_number]
        return train,validate

    def dataset_reset_back(self,target_list=None,label_setup=None):
        train_data = self.dataset_select_bylabel(self.train_data,target_list) if target_list is not None else self.train_data
        train_data = self.dataset_change_label(train_data,label_setup) if label_setup is not None else train_data
        validate_num = int(len(train_data)/5)
        training_num = int(len(train_data)-validate_num)
        train_data, validate_data = torch.utils.data.random_split(train_data, [training_num, validate_num])
        test_data = self.dataset_select_bylabel(self.test_data,target_list) if target_list is not None else self.test_data
        test_data = self.dataset_change_label(test_data,label_setup) if label_setup is not None else test_data
        return train_data,test_data,validate_data
    
    def dataset_reset(self,target_list=None,label_setup=None):
        def cell(target_list,label_setup,data):
            data = self.dataset_select_bylabel(data,target_list) if target_list is not None else data
            data = self.dataset_change_label(data,label_setup) if label_setup is not None else data
            return data
        
        train_data, validate_data = self.dataset_separate_validate(self.train_data)
        train_data = cell(target_list,label_setup,train_data)
        validate_data = cell(target_list,label_setup,validate_data)
        test_data = cell(target_list,label_setup,self.test_data)
        
        #train_data = self.dataset_select_bylabel(train_data,target_list) if target_list is not None else train_data
        #train_data = self.dataset_change_label(train_data,label_setup) if label_setup is not None else train_data
        
        #validate_data = self.dataset_select_bylabel(validate_data,target_list) if target_list is not None else validate_data
        #validate_data = self.dataset_change_label(validate_data,label_setup) if label_setup is not None else validate_data
        
        #test_data = self.dataset_select_bylabel(self.test_data,target_list) if target_list is not None else self.test_data
        #test_data = self.dataset_change_label(test_data,label_setup) if label_setup is not None else test_data
        return train_data,test_data,validate_data

