import os,sys,torch,copy
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets, domain=None, data_transforms=None,target_transforms=None,print_info=True):
        if targets is not None and len(data) != len(targets):
            raise Exception(f'data:{len(data)},targets:{len(targets)} not match!')
        
        self.data = data
        self.targets = targets
        self.domain = domain
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        if print_info: self.display_info()
    
    def display_info(self):
        print(f'Costom Dataset size: {[self.data.shape]}')
        if self.data_transforms is not None: print(f'Data Transforms: {self.data_transforms}')
        if self.target_transforms is not None: print(f'Target Transforms: {self.target_transforms}')
        print('='*100)

    def shuffle(self):
        idx = torch.randperm(len(self.data))
        self.data = self.data[idx]
        self.targets = self.targets[idx]
    
    def split(self, split_ratio=0.2):
        split_number = int(len(self.targets)*split_ratio)
        random_indices = torch.randperm(len(self))
        new_dataset = copy.deepcopy(self)
        part_1 = CustomDataset(data=new_dataset.data[random_indices[split_number:]],
                               targets=new_dataset.targets[random_indices[split_number:]],
                               data_transforms=self.data_transforms,
                               target_transforms=self.target_transforms)
        part_2 = CustomDataset(data=new_dataset.data[random_indices[:split_number]],
                               targets=new_dataset.targets[random_indices[:split_number]],
                               data_transforms=self.data_transforms,
                               target_transforms=self.target_transforms)
        return part_1, part_2

    def __len__(self): return len(self.data)

    def __getitem__(self, index):
        data = self.data[index] if self.data_transforms is None else self.data_transforms(self.data[index])
        if self.targets is None: return data
        target = self.targets[index] if self.target_transforms is None else self.target_transforms(self.targets[index])
        return data, target