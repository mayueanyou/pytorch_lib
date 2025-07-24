import os,sys,torch,copy
import numpy as np
from tqdm import tqdm
from . import*
from .functions import save_data,load_data
from .similarity_calculator import SimilarityCalculator
from .kmeans_calculator import KmeansCalculator


class LVB:
    def __init__(self,path, lvp_path_list=None,batch_mode=False) -> None:
        self.path = path
        self.lvb_file = f'{path}.npy'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sc = SimilarityCalculator()
        self.batch_mode = batch_mode
        self.lvp_path_list = lvp_path_list
        self.lvb_status = {'total_distance':0,
                           'distance_among_centroids':0,
                           'distance_centroids_between_mean':0,
                           'distance_pool_between_mean':0}
        
        self.load_lvps()
    
    def save(self):
        save_data(self.lvb_file,self.lvp_path_list)
    
    def load_lvps(self):
        if self.lvp_path_list is None:
            if os.path.isfile(self.lvb_file): self.lvp_path_list = load_data(self.lvb_file)
            else: return
        
        self.lvp_list = [LVP(path) for path in self.lvp_path_list]
        self.total_class = len(self.lvp_list)
    
    def save_lvps(self):
        for lvp in self.lvp_list: lvp.save()
    
    def initialize_lvp_from_dataset(self,dataset,target_list=None):
        lvp_path_list = []
        if target_list is None: target_list = range(torch.max(dataset.targets)+1)
        for i in target_list:
            data_i = dataset.data[dataset.targets==i]
            path = f'{self.path}_{i}.pt'
            lvp_i = LVP(path= path, generate=True,id = i,data = data_i)
            lvp_i.save()
            lvp_path_list.append(path)
        self.lvp_path_list = np.array(lvp_path_list)
        
        save_data(self.lvb_file,self.lvp_path_list)
        self.load_lvps()

    def generate_confidence(self):
        max_values, max_indices = torch.min(self.var_bank, dim=1, keepdim=True)
        self.lv_confidence = torch.ones(self.lv_bank.shape[0],self.lv_bank.shape[1])
        #self.lv_confidence = 1 - self.var_bank/max_values
        #self.lv_bank *= self.lv_confidence
    
    def generate_lvb_from_dataset(self,dataset,k=10,k_range=None,target_list=None,dis_func='L1',from_mean_rate=1):
        self.initialize_lvp_from_dataset(dataset,target_list)
        self.generate_lvb(k=k,k_range=k_range,dis_func=dis_func,from_mean_rate=from_mean_rate)
        self.save_lvps()
        self.save()

    def generate_lvb(self,k=10,k_range=None,dis_func='L1',from_mean_rate=1):
        for lvp in tqdm(self.lvp_list):
            lvp.generate_k_cluster(k=k,k_range=k_range,dis_func=dis_func,from_mean_rate=from_mean_rate)
    
    def get_lvb(self,mode='m'):
        lv_list, var_list, id_list = [], [], []
        for lvp in self.lvp_list:
            lv,var,id =  lvp.get_lvp(mode)
            lv_list.append(lv)
            if var is not None: var_list.append(var)
            id_list.append(id)
            
        self.lv_bank = torch.cat((lv_list))
        self.var_bank = None if len(var_list)==0 else torch.cat((var_list))
        self.id_bank = torch.cat((id_list))
        self.lvb_size = self.lv_bank.shape
        print(f'LVB: mode {mode} class {self.total_class} size {self.lvb_size}')
    
    def eval_dataset(self,dataset,mode='m',dis_func='L1',with_confidence=False):
        self.get_lvb(mode)
        self.id_bank = self.id_bank.to(self.device)
        
        if not with_confidence:
            values, indices, similarity, similarity_raw = self.sc(self.lv_bank,dataset.data,dis_func=dis_func,batch_mode=self.batch_mode)
            indices = torch.flatten(indices)
            indices = self.id_bank[indices]
            acc = torch.sum(torch.eq(dataset.targets.to(self.device),indices.to(self.device))) / len(dataset.targets)
            
        else:
            self.generate_confidence()
            for image_features, labels in tqdm(dataloader):
                image_features = image_features.unsqueeze(1)
                image_features = image_features.repeat(1,len(self.lv_bank),1)
                #image_features = image_features * self.lv_confidence
                distence = torch.abs(image_features-self.lv_bank)
                
                var = self.var_bank.unsqueeze(0)
                var = var.repeat(len(image_features),1,1)
                #distence[distence < var] *= 1.1 #distence[distence > var] / var[distence > var]
                distence[var==torch.zeros(768)] = 100

                distence = torch.sum(distence,dim=2) #/ torch.count_nonzero(distence,dim=2)
                
                indices = torch.argmin(distence, dim=1)
                indices = self.id_bank[indices]
                result = torch.eq(labels.to(self.device),indices.to(self.device))
                acc += torch.sum(result)
                num_data += len(labels)
        
        print('accuracy: ',acc)
        return acc

    def extend(self,lv,id):
        self.lv_bank = torch.cat((self.lv_bank,lv))
        self.id_bank = torch.cat((self.id_bank,id))
        self.lvb_size = self.lv_bank.shape
    
    def select_by_label(self,target_list):
        if target_list is None: return
        idx = sum(self.id_bank==i for i in target_list).bool()
        self.lv_bank = self.lv_bank[idx]
        self.id_bank = self.id_bank[idx]
        print(f'select label:\n {target_list}')
        self.lvb_size = self.lv_bank.shape

class LVP:
    def __init__(self,
                 path,
                 generate=False,
                 pool_size_limit = 100,
                 id=None,
                 data=None,
                 print_info=False,
                 remove_amount=10) -> None:
        
        self.file_path = path
        self.pool_size_limit = pool_size_limit
        self.kc = KmeansCalculator()
        self.sc = SimilarityCalculator()
        self.remove_amount = remove_amount
        
        if generate:
            self.id = id
            self.lvp = {}
            self.add_lvp('p',data)
            self.add_lvp('m',torch.unsqueeze(torch.mean(data,dim=0),dim=0),torch.unsqueeze(torch.var(data,dim=0),dim=0))
            if self.remove_amount > 0: self.remove_far_points()
        elif os.path.isfile(self.file_path): self.load()

        if print_info: print(f'LVP: id: [{self.id}] pool size: {self.lvp["p"]["data"].shape}')

    def add_lvp(self,name,data,data_var=None,info=None):
        self.lvp[name] = {'data': data, 'var': data_var, 'info': info}
    
    def remove_far_points(self):
        values, indices, similarity, similarity_raw = self.sc(self.lvp['p']['data'],self.lvp['m']['data'],topk=len(self.lvp['p']['data']),dis_func='L1')
        indices = indices.flatten().cpu()
        self.lvp['p']['data'] = self.lvp['p']['data'][indices[:-self.remove_amount]]

    def load(self):
        data = torch.load(self.file_path)
        self.id = data['id']
        self.lvp = data['lvp']
    
    def save(self):
        data = {'id':self.id,'lvp':self.lvp}
        save_data(self.file_path,data)
    
    def get_lvp(self,mode='m'):
        lvp = self.lvp[mode]['data']
        lvp_var = self.lvp[mode]['var']
        if mode not in ['m','mean','p','km','itkm']: raise Exception("mode is not exist!")

        lvp_id = torch.tensor([self.id]*len(lvp),dtype=torch.long)
        return lvp, lvp_var, lvp_id
        
    
    # def extend(self,new_lvps):
    #     self.lvp = torch.cat((self.lvp,new_lvps))
    #     if len(self.lvp) > self.pool_size:
    #         self.generate_k_cluster(self.pool_size)
    #         self.lvp = self.lvp_clusters
    #     self.lvp_mean = torch.unsqueeze(torch.mean(self.lvp,dim=0),dim=0)
    #     self.lvp_var = torch.unsqueeze(torch.var(self.lvp,dim=0),dim=0)
    #     #self.lvp_confidence = 1 - self.lvp_var/torch.max(self.lvp_var)
    
    def generate_k_cluster(self,k=10,k_range=None,max_iterations=100,dis_func='L1', from_mean_rate=1):
        def generate_var(centroids,labels):
            clusters_var = []
            for i in range(len(centroids)): 
                centroid_group = self.lvp['p']['data'][labels == i]
                if len(centroid_group) == 1: centroid_var = torch.zeros(1,centroids.shape[-1])
                else: centroid_var = torch.unsqueeze(torch.var(centroid_group, dim=0),dim=0)
                clusters_var.append(centroid_var)
            clusters_var = torch.cat((clusters_var))
            return clusters_var
        
        print('='*100)
        print(f'Generating k cluster: id: {self.id} k: {k}')
        self.k = k
        self.kc.group_dis_func = dis_func
        self.kc.from_mean_rate = from_mean_rate
        
        centroids,labels = self.kc(self.lvp['p']['data'],centroids=k,max_iterations=max_iterations)
        clusters_var = generate_var(centroids,labels)
        self.add_lvp('km',centroids,clusters_var,info={'k':k,'max_iterations':max_iterations})
        
        if k_range is not None: 
            print('='*100)
            print(f'Generating iterative k cluster: id: {self.id} k: {k} k_range: {k_range}')
            centroids,labels = self.kc.iterative_generation(data_pool=self.lvp['p']['data'],k=k,k_range=k_range,max_iterations=max_iterations)
            clusters_var = generate_var(centroids,labels)
            self.add_lvp('itkm',centroids,clusters_var,info={'k':k,'k_range':k_range,'max_iterations':max_iterations})