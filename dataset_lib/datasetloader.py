import os,sys,copy,torch,random,cv2,torchvision,json
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF
from torchvision import datasets,transforms


class CustomDataset(Dataset):
    def __init__(self, data, targets, normalize = False,data_transform=None,target_transform=None):
        self.data = data
        self.targets = targets
        if normalize: self.normalize()
        self.data_shape = data.shape
        self.targets_shape = targets.shape
        self.data_transform = data_transform
        self.target_transform = target_transform
        #print(f'data shape: {self.data_shape}')
        #print(f'targets shape: {self.targets_shape}')
    
    def normalize(self):
        self.data = torch.div(self.data, torch.max(self.data,dim=0).values)
        self.targets = torch.div(self.targets, torch.max(self.targets,dim=0).values)    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_transform is not None: self.data[index] = self.data_transform(self.data[index])
        if self.target_transform is not None: self.targets[index] = self.target_transform(self.targets[index])
        return self.data[index], self.targets[index]

class CustomDatasetLoader:
    def __init__(self,train_data,test_data,validate_data,target_list=None,label_setup=None,batch_size=64) -> None:
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
    def __init__(self,train_data,test_data,validate_data=None) -> None:
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

