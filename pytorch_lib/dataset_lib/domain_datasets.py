import os,sys,wget,zipfile,pathlib,shutil
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor,Resize,Compose

def get_classes_from_file(path):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + path, 'r') as file: classes_original = json.loads(file.read())
        classes = list(classes_original.values())
        return classes

class OfficeHome:
    def __init__(self,dataset_path,data_transform=Compose([Resize((224,224)),ToTensor()])) -> None:
        self.name = type(self).__name__
        if not pathlib.Path(dataset_path + '/OfficeHomeDataset_10072016.zip').is_file():
            print(dataset_path + '/OfficeHomeDataset_10072016.zip','is not exist')
            return
        
        if not pathlib.Path(dataset_path + '/extracted').is_file():
            with zipfile.ZipFile(dataset_path + '/OfficeHomeDataset_10072016.zip',"r") as zip_ref: zip_ref.extractall(dataset_path)
            with open(dataset_path + '/extracted', "w") as f:...
        
        self.dataset_path = dataset_path
        self.data_transform = data_transform
        
        self.domain_list = ['Art','Clipart','Product','Real World']
        self.classes = get_classes_from_file(f'/supplement/{self.name}/classes.txt')
        #self.classes = sorted(self.classes)
        for i in range(len(self.classes)): self.classes[i] = self.classes[i].lower()
        
        self.datasets = []
        for domain_name in self.domain_list:
            self.datasets.append(datasets.ImageFolder(root = dataset_path + f'/OfficeHomeDataset_10072016/{domain_name}',transform = data_transform))
        
        
    def get_loaders(self,batch_size=64):
        loaders = []
        for dataset in self.datasets:
            loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
        return loaders

class DomainNet:
    def __init__(self,root,data_transform=Compose([Resize((224,224)),ToTensor()])) -> None:
        self.name = type(self).__name__
        self.root = root
        self.zip_file_list = [['/clipart.zip', 'https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip'],
                              ['/infograph.zip', 'https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip'],
                              ['/painting.zip', 'https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip'],
                              ['/quickdraw.zip', 'https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip'],
                              ['/real.zip', 'https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip'],
                              ['/sketch.zip', 'https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip']]
        
        self.train_txt_file_list = [['/clipart_train.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt'],
                              ['/infograph_train.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt'],
                              ['/painting_train.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt'],
                              ['/quickdraw_train.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt'],
                              ['/real_train.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt'],
                              ['/sketch_train.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt']]
        
        self.test_txt_file_list = [['/clipart_test.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt'],
                              ['/infograph_test.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt'],
                              ['/painting_test.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt'],
                              ['/quickdraw_test.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt'],
                              ['/real_test.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt'],
                              ['/sketch_test.txt', 'https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt']]
        
        self.domain_list = ['clipart','infograph','painting','quickdraw','real','sketch']
        self.classes_folders = get_classes_from_file(f'/supplement/{self.name}/classes.txt')
        
        self.classes = [it.replace('_',' ') for it in self.classes_folders]
        self.classes = sorted(self.classes)
        
        self.check_file(self.zip_file_list)
        self.check_file(self.train_txt_file_list)
        self.check_file(self.test_txt_file_list)
        self.extracte_file()
        self.split_dataset()
        
        self.train_datasets = []
        self.test_datasets = []
        for domain_name in self.domain_list:
            self.train_datasets.append(datasets.ImageFolder(root = self.root + f'/train/{domain_name}',transform = data_transform))
            self.test_datasets.append(datasets.ImageFolder(root = self.root + f'/test/{domain_name}',transform = data_transform))
    
    def get_loaders(self,batch_size=64):
        train_loaders,test_loaders = [],[]
        for dataset in self.train_datasets: train_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
        for dataset in self.test_datasets: test_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
        return train_loaders,test_loaders
    
    def check_file(self,file_list):
        for file in file_list: 
            if not pathlib.Path(self.root + file[0]).is_file(): 
                print('downing: ', file[0])
                wget.download(url = file[1],out = self.root + file[0])
    
    def extracte_file(self):
        if not pathlib.Path(self.root + '/extracted').is_file():
            for file in self.zip_file_list:
                print('extracting: ',file[0])
                with zipfile.ZipFile(self.root + file[0],"r") as zip_ref: zip_ref.extractall(self.root)
            with open(self.root + '/extracted', "w") as f:...
    
    def split_dataset(self):
        def move_file(file_path,folder_name):
            with open(file_path,'r') as file:
                lines = file.readlines()
            lines = [it.replace('\n','') for it in lines]
            lines = [it.split() for it in lines]
            lines = [it[0] for it in lines]
            for line in lines: shutil.copy(self.root + '/' + line, self.root + folder_name + '/' + line)
        
        if not pathlib.Path(self.root + '/splited').is_file():
            for name in self.classes_folders:
                for domain_name in self.domain_list:
                    if not os.path.exists(self.root + f'/train/{domain_name}/{name}'):os.makedirs(self.root + f'/train/{domain_name}/{name}')
                    if not os.path.exists(self.root + f'/test/{domain_name}/{name}'):os.makedirs(self.root + f'/test/{domain_name}/{name}')
            
            for file in self.train_txt_file_list: move_file(self.root + file[0],'/train')
            for file in self.test_txt_file_list: move_file(self.root + file[0],'/test')
            with open(self.root + '/splited', "w") as f:...

class CORe50:
    def __init__(self,root,data_transform=Compose([Resize((224,224)),ToTensor()])) -> None:
        self.name = type(self).__name__
        self.root = root
        if not pathlib.Path(root).is_dir():
            print(root,'is not exist')
            return
        self.train_domain_list = ['s1','s2','s4','s5','s6','s8','s9','s11']
        self.test_domain_list = ['s3','s7','s10']
        self.classes = ['plug adapters']*5 + ['mobile phones']*5 + ['scissors']*5 + ['light bulbs']*5 + ['cans']*5 + \
                        ['glasses']*5 + ['balls']*5 + ['markers']*5 + ['cups']*5 + ['remote controls']*5 
        #print( self.classes)
        
        self.train_datasets = []
        self.test_datasets = []
        for domain in self.train_domain_list:
            self.train_datasets.append(datasets.ImageFolder(root = self.root + f'/{domain}',transform = data_transform))
        
        for domain_name in self.test_domain_list:
            self.test_datasets.append(datasets.ImageFolder(root = self.root + f'/{domain_name}',transform = data_transform))
        
    def get_loaders(self,batch_size=64):
        train_loaders,test_loaders = [],[]
        for dataset in self.train_datasets: train_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
        for dataset in self.test_datasets: test_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
        return train_loaders,test_loaders

if __name__ == "__main__":
    #dn = DomainNet('/home/yma183/datasets/DomainNet')
    #oh = OfficeHome('/home/yma183/datasets/OfficeHome')
    co = CORe50('/home/yma183/datasets/CORe50/core50/data/core50_128x128')