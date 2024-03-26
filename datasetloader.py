import os,sys,copy,torch,random,cv2,torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF

class CustomDataset(Dataset):
    def __init__(self, data, targets, normalize = False):
        self.data = data
        self.targets = targets
        if normalize: self.normalize()
        self.data_shape = data[0].shape
        self.targets_shape = targets[0].shape
        print(f'data shape: {self.data_shape}')
        print(f'targets shape: {self.targets_shape}')
    
    def normalize(self):
        self.data = torch.div(self.data, torch.max(self.data,dim=0).values)
        self.targets = torch.div(self.targets, torch.max(self.targets,dim=0).values)    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

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
        print(f'data per batch: train[{len(train)//batch_size}] test[{len(test)//batch_size}] validate[{len(validate)//batch_size}]\n')
    
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
        train_data, validate_data = self.dataset_separate_validate(self.train_data)
        
        train_data = self.dataset_select_bylabel(train_data,target_list) if target_list is not None else train_data
        train_data = self.dataset_change_label(train_data,label_setup) if label_setup is not None else train_data
        
        validate_data = self.dataset_select_bylabel(validate_data,target_list) if target_list is not None else validate_data
        validate_data = self.dataset_change_label(validate_data,label_setup) if label_setup is not None else validate_data
        
        test_data = self.dataset_select_bylabel(self.test_data,target_list) if target_list is not None else self.test_data
        test_data = self.dataset_change_label(test_data,label_setup) if label_setup is not None else test_data
        return train_data,test_data,validate_data

