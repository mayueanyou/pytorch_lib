import os,sys,json,pathlib,shutil,torch
import pandas as pd
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Compose,Resize
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path

from .dataset import DatasetLoader,CustomDataset
from ..utility import save_dataset_images
from ..transform_lib.transform import COCOTargetsTF

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
    def get_datas(self):pass
    
    @abstractmethod
    def get_loaders(self):pass
    
    def classes(self):
        return self.classes

class ImageFolder:
    def __init__(self,dataset_path:str,transform=ToTensor()) -> None:
        self.root = dataset_path
        self.data = datasets.ImageFolder(root = self.root ,transform = transform)
    
    def get_loaders(self,batch_size=64):
        dataloader = DataLoader(self.data, batch_size = batch_size)

        return dataloader

class MNIST:
    def __init__(self,dataset_path:str,training_transform=ToTensor(),test_transform=ToTensor(),load_data=True) -> None:
        if load_data: self.load(dataset_path,training_transform,test_transform)
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.name = 'MNIST'
        print('Dataset: ',self.name)
    
    def load(self,dataset_path,training_transform,test_transform):
        self.train_dataset = datasets.MNIST(root=dataset_path,train=True,download=True,transform=training_transform)
        self.test_dataset = datasets.MNIST(root=dataset_path,train=False,download=True,transform=test_transform)
        self.dataset_loader = DatasetLoader(self.train_dataset,self.test_dataset)
    
    def get_datas(self,target_list=None,labels_setup=None):
        return self.dataset_loader.get_datas(target_list=target_list,labels_setup=labels_setup)
    
    def get_loaders(self,target_list=None,labels_setup=None,batch_size=64):
        return self.dataset_loader.get_loaders(target_list=target_list,labels_setup=labels_setup,batch_size=batch_size)

class CIFAR10(VisionDataset):
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),load_data=True) -> None:
        super().__init__()
        self.root = dataset_path
        if load_data: self.load(dataset_path,training_transform,test_transform)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.name = 'CIFAR10'
        print('Dataset: ',self.name)
    
    def load(self,dataset_path,training_transform,test_transform):
        self.train_dataset =datasets.CIFAR10(root=dataset_path,train=True,download=True,transform=training_transform)
        self.test_dataset = datasets.CIFAR10(root=dataset_path,train=False,download=True,transform=test_transform)
        self.dataset_loader = DatasetLoader(self.train_dataset,self.test_dataset)
    
    def get_datas(self,target_list=None,label_setup=None):
        return self.dataset_loader.get_datas(target_list=target_list,label_setup=label_setup)
    
    def get_loaders(self,target_list=None,label_setup=None,batch_size=64):
        return self.dataset_loader.get_loaders(target_list=target_list,label_setup=label_setup,batch_size=batch_size)
    
    def save_images(self,path=None):
        if path is None: path = self.root
        save_dataset_images(self.train_dataset,path+'/train','cifar10')
        save_dataset_images(self.test_dataset,path+'/test','cifar10')

class CIFAR100(VisionDataset):
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),load_data=True) -> None:
        super().__init__()
        self.root = dataset_path
        if load_data: self.load(dataset_path,training_transform,test_transform)
        print('Dataset: ',self.name)
        self.get_classes_from_file()
    
    def load(self,dataset_path,training_transform,test_transform):
        self.train_dataset = datasets.CIFAR100(root=dataset_path,train=True,download=True,transform=training_transform)
        self.test_dataset = datasets.CIFAR100(root=dataset_path,train=False,download=True,transform=test_transform)
        self.dataset_loader = DatasetLoader(self.train_dataset,self.test_dataset)

    def get_datas(self,target_list=None,label_setup=None):
        return self.dataset_loader.get_datas(target_list=target_list,label_setup=label_setup)
    
    def get_loaders(self,target_list=None,label_setup=None,batch_size=64):
        return self.dataset_loader.get_loaders(target_list=target_list,label_setup=label_setup,batch_size=batch_size)
    
    def save_images(self,path=None):
        if path is None: path = self.root
        save_dataset_images(self.train_dataset,path+'/train','cifar100')
        save_dataset_images(self.test_dataset,path+'/test','cifar100')
    
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

class COCO:
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor()) -> None:
        self.root = dataset_path
        self.train_dataset = datasets.CocoDetection(root=self.root + '/train2017', annFile=self.root + '/annotations/instances_train2017.json', transform=training_transform)
        self.test_dataset = datasets.CocoDetection(root=self.root + '/val2017', annFile=self.root + '/annotations/instances_val2017.json', transform=test_transform)
        self.name = 'COCO'
        print('Dataset: ',self.name)
    
    def get_loaders(self,batch_size=64):
        train_dataloader = DataLoader(self.train_dataset, batch_size = batch_size,collate_fn=lambda batch: list(zip(*batch)))
        test_dataloader = DataLoader(self.test_dataset, batch_size = batch_size, collate_fn=lambda batch: list(zip(*batch)))
        return train_dataloader,test_dataloader

    def wrap_dataset_for_transforms(self):
        self.train_dataset = datasets.wrap_dataset_for_transforms_v2(self.train_dataset, target_keys=["boxes", "labels", "masks"])
        self.test_dataset = datasets.wrap_dataset_for_transforms_v2(self.test_dataset, target_keys=["boxes", "labels", "masks"])

def coco_targets_transforms(annotations):
    print(annotations)
    input()
    masks = []
    boxes = []
    labels = []
    for annotation in annotations:
        if 'boxes' in annotation:
            boxes.append(torch.tensor(annotation['boxes'], dtype=torch.float32))
        else:
            boxes.append(torch.tensor([], dtype=torch.float32))
        
        if 'labels' in annotation:
            labels.append(torch.tensor(annotation['labels'], dtype=torch.int64))
        else:
            labels.append(torch.tensor([], dtype=torch.int64))
        
        if 'masks' in annotation:
            masks.append(torch.tensor(annotation['masks'], dtype=torch.bool))
        else:
            masks.append(torch.tensor([], dtype=torch.bool))
    
    return {'boxes': boxes, 'labels': labels, 'masks': masks}

class COCOCustom:
    def __init__(self,dataset_path,training_transform=ToTensor(),test_transform=ToTensor(),image_size=(480, 640)) -> None:
        self.root = dataset_path
        self.preprocess_annotations()
        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []
        for key in self.train_annotations.keys():
            self.train_data += [self.root + '/train2017/' + key]
            self.train_targets.append(self.train_annotations[key]['annotations'])
        for key in self.val_annotations.keys():
            self.test_data += [self.root + '/val2017/' + key]
            self.test_targets.append(self.val_annotations[key]['annotations'])
        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)
        
        # key = '000000397133.jpg'
        # print(self.val_annotations[key]['annotations'])
        # print(len(self.val_annotations[key]['annotations']))
        # print()
        # input()
        
        self.train_dataset = CustomDataset(data=self.train_data, targets=self.train_targets,data_transforms=training_transform,targets_transforms=COCOTargetsTF(image_size[0], image_size[1]))
        self.test_dataset = CustomDataset(data=self.test_data, targets=self.test_targets,data_transforms=test_transform,targets_transforms=COCOTargetsTF(image_size[0], image_size[1]))
        self.name = 'COCOCustom'
        print('Dataset: ',self.name)
    
    def preprocess_annotations(self):
        def process_cell(original_annotations,subset):
            new_annotations = {}
            for i in range(len(original_annotations['images'])):
                file_name = original_annotations['images'][i]['file_name']
                if not os.path.isfile(os.path.join(self.root + f'/{subset}2017', file_name)): continue
                width = original_annotations['images'][i]['width']
                height = original_annotations['images'][i]['height']
                new_annotations[file_name] = {'annotations': [], 'width': width, 'height': height}

            for i in tqdm(range(len(original_annotations['annotations']))):
                file_name = f'{original_annotations["annotations"][i]["image_id"]:012d}.jpg'
                if file_name not in new_annotations: continue
                
                annotations = {}
                height = new_annotations[file_name]['height']
                width = new_annotations[file_name]['width']
                annotations['labels'] = [original_annotations['annotations'][i]['category_id']]
                annotations['height'] = height
                annotations['width'] = width

                box = original_annotations['annotations'][i]['bbox']
                box[0], box[1], box[2], box[3] = box[0] / width, box[1] / height, box[2] / width, box[3] / height
                annotations['boxes'] = box

                annotations['segmentation'] = original_annotations['annotations'][i]['segmentation']
                
                new_annotations[file_name]['annotations'].append(annotations)
            return new_annotations

        #if True:
        if not os.path.isfile(self.root + '/annotations_2017.json'):
            print('Processing annotations...')
            with open(self.root + '/annotations/instances_train2017.json', 'r') as file: train_annotations_original = json.load(file)
            with open(self.root + '/annotations/instances_val2017.json', 'r') as file: val_annotations_original = json.load(file)
            self.train_annotations = process_cell(train_annotations_original,'train')
            self.val_annotations = process_cell(val_annotations_original,'val')
            json.dump({'train': self.train_annotations, 'val': self.val_annotations}, open(self.root + '/annotations_2017.json', 'w'))
            print(f'Annotations saved to {self.root}/annotations_2017.json')

        else:
            with open(self.root + '/annotations_2017.json', 'r') as file: annotations = json.load(file)
            self.train_annotations = annotations['train']
            self.val_annotations = annotations['val']
            print(f'Annotations loaded from {self.root}/annotations_2017.json')
    
    def get_loaders(self,batch_size=64):
        train_dataloader = DataLoader(self.train_dataset, batch_size = batch_size,collate_fn=lambda batch: list(zip(*batch)))
        test_dataloader = DataLoader(self.test_dataset, batch_size = batch_size, collate_fn=lambda batch: list(zip(*batch)))
        return train_dataloader,test_dataloader