import os,sys,copy,torch
from torch.utils.data import DataLoader

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