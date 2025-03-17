import os,sys,copy,torch,random,cv2,torchvision,json,pathlib,toml,yaml
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF
from torchvision import datasets,transforms
from abc import ABC, abstractmethod


class VisionDataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
    
    def get_classes_from_file(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + f'/supplement/{self.name}/classes.txt', 'r') as file: self.classes_original = json.loads(file.read())
        self.classes = list(self.classes_original.values())
        #print(f'classes: {self.classes}')
        
        #with open(current_path + f'/supplement/{self.name}/classes.toml', 'r') as file: self.classes_original = toml.load(file)
        #class_name = [i[1] for i in self.classes_original['class']]
        #print(f'classes: {class_name}')
    
    @abstractmethod
    def load(self):pass
    
    @abstractmethod
    def datas(self):pass
    
    @abstractmethod
    def loaders(self):pass
    
    def classes(self):
        return self.classes


class ImageFolderDataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__



def get_classes_from_file(path):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + path, 'r') as file: classes_original = json.loads(file.read())
        classes = list(classes_original.values())
        return classes

class CustomDataset(Dataset):
    def __init__(self, data, targets, data_transforms=None,target_transforms=None):
        if len(data) != len(targets): raise Exception(f'data:{len(data)},targets:{len(targets)} not match!')
        
        self.data = data
        self.targets = targets
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        print(f'Costom Dataset: {len(self.data)}')
        print(f'Data Transforms: {self.data_transforms}')
        print(f'Target Transforms: {self.target_transforms}')
        print('='*100)

    def shuffle(self):
        idx = torch.randperm(len(self.data))
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __len__(self): return len(self.data)

    def __getitem__(self, index):
        data = self.data[index] if self.data_transforms is None else self.data_transforms(self.data[index])
        target = self.targets[index] if self.target_transforms is None else self.target_transforms(self.targets[index])
        return data, target

class DatasetWrapper:
    def __init__(self,dataset) -> None:
        self.dataset = copy.deepcopy(dataset)
        self.dataset_out = copy.deepcopy(self.dataset)
        self.length_original = len(dataset)
        self.length_out = len(self.dataset_out)
        if not torch.is_tensor(self.dataset.targets): self.dataset.targets = torch.tensor(self.dataset.targets)
    
    def __len__(self): return len(self.dataset_out)
    
    def reset(self): 
        self.dataset_out = copy.deepcopy(self.dataset)
        self.length_original = len(self.dataset)
        self.length_out = len(self.dataset_out)
    
    def extend(self,dataset):
        self.dataset.data = torch.cat([self.dataset.data,dataset.data])
        self.dataset.targets = torch.cat([self.dataset.targets,dataset.targets])
        self.reset()
    
    def get_sample(self,target_list=None,amount=1):
        if target_list is None: target_list = list(range(max(self.dataset_out.targets)))
        data_list = []
        targets_list = []
        for target in target_list:
            data_list.append(self.dataset_out.data[self.dataset_out.targets==target][:amount])
            max_amount = max(len(data_list[-1]),amount)
            targets_list.append(torch.tensor([target]*max_amount))
        data = torch.cat(data_list)
        targets = torch.cat(targets_list)
        return data,targets
    
    def select_bylabel(self,target_list):
        idx = sum(self.dataset_out.targets==i for i in target_list).bool()
        self.dataset_out.data = self.dataset_out.data[idx]
        self.dataset_out.targets = self.dataset_out.targets[idx]
        print(f'select label:\n {target_list}')
    
    def change_label(self,label_setup):
        for setup in label_setup:
            idx = sum(self.dataset_out.targets==i for i in setup[0]).bool()
            self.dataset_out.targets[idx] = setup[1]
        print('change label:')
        for it in label_setup: print(f'{it[0]} -> {it[1]}')
    
    def split(self,rate=0.2):
        split_number = int(self.length_original*rate)
        random_indices = torch.randperm(self.length_original)
        part_1 = copy.deepcopy(self.dataset)
        part_1.data = part_1.data[random_indices[split_number:]]
        part_1.targets = part_1.targets[random_indices[split_number:]]
        part_2 = copy.deepcopy(self.dataset)
        part_2.data = part_2.data[random_indices[:split_number]]
        part_2.targets = part_2.targets[random_indices[:split_number]]
        print(f'split: part_1[{len(part_1)}] part_2[{len(part_2)}]')
        return DatasetWrapper(part_1),DatasetWrapper(part_2)
    
    def transform(self,target_list=None,label_setup=None):
        self.reset()
        self.dataset_out = copy.deepcopy(self.dataset)
        if target_list is not None: self.select_bylabel(target_list)
        if label_setup is not None: self.change_label(label_setup)
        self.length_out = len(self.dataset_out)
    
    def __call__(self,batch_size=64,shuffle=False):
        return DataLoader(self.dataset_out, batch_size = batch_size,shuffle=shuffle)

class DatasetLoader():
    def __init__(self,train_data,test_data=None,validate_data=None,generate_validate=True,validate_rate=0.2) -> None:
        self.validate_rate = validate_rate
        self.train_data = DatasetWrapper(train_data) if train_data is not None else None
        self.test_data = DatasetWrapper(test_data) if test_data is not None else None
        self.validate_data = DatasetWrapper(validate_data) if validate_data is not None else None
        if validate_data is None and generate_validate:self.train_data,self.validate_data =  self.train_data.split(self.validate_rate)
    
    def print_info(self,train,test,validate,batch_size):
        print(f'batch size: {batch_size}')
        batch_size = abs(batch_size)
        print(f'data in total:  train[{len(train)}] test[{len(test)}] validate[{len(validate)}]')
        print(f'batchs in total: train[{len(train)//batch_size}] test[{len(test)//batch_size}] validate[{len(validate)//batch_size}]\n')
    
    def get_datas(self,target_list=None,label_setup=None):
        self.dataset_reset(target_list,label_setup)
        return self.train_data.dataset_out, self.test_data.dataset_out, self.validate_data.dataset_out
    
    def get_loaders(self,target_list=None,label_setup=None,batch_size = 64,shuffle=False):
        self.dataset_reset(target_list,label_setup)
        self.print_info(self.train_data, self.test_data, self.validate_data,batch_size)
        if batch_size == -1:
            train_dataloader = self.train_data(self.train_data.length,shuffle=shuffle)
            test_dataloader = self.test_data(self.test_data.length,shuffle=shuffle)
            validate_dataloader = self.validate_data(self.validate_data.length,shuffle=shuffle)
        else:
            train_dataloader = self.train_data(batch_size,shuffle=shuffle)
            test_dataloader = self.test_data(batch_size,shuffle=shuffle)
            validate_dataloader = self.validate_data(batch_size,shuffle=shuffle)
        return train_dataloader,test_dataloader,validate_dataloader
    
    def dataset_reset(self,target_list=None,label_setup=None):
        if self.train_data is not None: self.train_data.transform(target_list=target_list,label_setup=label_setup)
        if self.test_data is not None: self.test_data.transform(target_list=target_list,label_setup=label_setup)
        if self.validate_data is not None: self.validate_data.transform(target_list=target_list,label_setup=label_setup)

class DatasetLoader_old():
    def __init__(self,train_data:Dataset,test_data,validate_data=None) -> None:
        self.validate_rate = 0.2
        self.train_data = train_data
        if not torch.is_tensor(self.train_data.targets): self.train_data.targets = torch.tensor(self.train_data.targets)
        self.test_data = test_data
        if not torch.is_tensor(self.test_data.targets): self.test_data.targets = torch.tensor(self.test_data.targets)
        if validate_data is not None:
            self.validate_data = validate_data
            if not torch.is_tensor(self.validate_data.targets): self.validate_data.targets = torch.tensor(self.validate_data.targets)
        else:
            self.train_data,self.validate_data =  self.dataset_separate_validate(self.train_data)
    
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
        
        train_data = cell(target_list,label_setup,self.train_data)
        validate_data = cell(target_list,label_setup,self.validate_data)
        test_data = cell(target_list,label_setup,self.test_data)
        return train_data,test_data,validate_data

