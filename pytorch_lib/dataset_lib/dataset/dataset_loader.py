import os,sys,copy,torch,random,cv2,torchvision,json,pathlib,toml,yaml
from .dataset_wrapper import DatasetWrapper

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
