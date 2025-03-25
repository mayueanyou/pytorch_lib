import os,sys,torch
from torch.utils.data import DataLoader

from .dataset import CustomDataset
from .dataset  import DatasetWrapper
from .dataset.functions import load_dataset_from_yaml

class YamlDataset():
    def __init__(self,yaml_path,data_transforms=None,target_transforms=None) -> None:
        self.name = type(self).__name__
        dataset_raw = load_dataset_from_yaml(yaml_path)
        self.dataset = CustomDataset(dataset_raw['data'],torch.tensor(dataset_raw['targets'],dtype=torch.long),
                                    data_transforms=data_transforms,target_transforms=target_transforms)
    
    def get_datas(self):
        return self.dataset
    
    def get_loader(self, batch_size = 64, shuffle = False):
        return DataLoader(self.dataset, batch_size = batch_size,shuffle=shuffle)