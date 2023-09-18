import os,sys,copy,torch,random,cv2,torchvision
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

class DatasetLoader():
    def __init__(self,train_data,test_data,batch_size = 64) -> None:
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_data.targets = torch.tensor(self.train_data.targets)
        self.test_data = test_data
        self.test_data.targets = torch.tensor(self.test_data.targets)
    
    def print_info(self,train,test,validate):
        print(f'batch size: {self.batch_size}')
        print(f'data in total:  train[{len(train)}] test[{len(test)}] validate[{len(validate)}]')
        print(f'data per batch: train[{len(train)/self.batch_size}] test[{len(test)/self.batch_size}] validate[{len(validate)/self.batch_size}]')
    
    def get_loaders(self,target_list=None,label_setup=None):
        training_data,test_data,validate_data = self.dataset_reset(target_list,label_setup)
        self.print_info(training_data,test_data,validate_data)
        train_dataloader = DataLoader(training_data, batch_size = self.batch_size)
        test_dataloader = DataLoader(test_data, batch_size = self.batch_size)
        validate_dataloader = DataLoader(validate_data, batch_size = self.batch_size)
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

    def dataset_reset(self,target_list=None,label_setup=None):
        train_data = self.dataset_select_bylabel(self.train_data,target_list) if target_list is not None else self.train_data
        train_data = self.dataset_change_label(train_data,label_setup) if label_setup is not None else train_data
        validate_num = int(len(train_data)/5)
        training_num = int(len(train_data)-validate_num)
        train_data, validate_data = torch.utils.data.random_split(train_data, [training_num, validate_num])
        test_data = self.dataset_select_bylabel(self.test_data,target_list) if target_list is not None else self.test_data
        test_data = self.dataset_change_label(test_data,label_setup) if label_setup is not None else test_data
        return train_data,test_data,validate_data

class RGB_Add_Gray:
    def __init__(self) -> None:
        pass

    def __call__(self, pic):
        pic = F.to_tensor(pic)
        gray_pic = F.rgb_to_grayscale(pic)
        return torch.cat((pic,gray_pic))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"