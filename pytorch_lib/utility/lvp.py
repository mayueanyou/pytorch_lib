import os,sys,torch,copy
from tqdm import tqdm
from . import*
from .functions import k_means,save_tensor
from .similarity_calculator import SimilarityCalculator

class LVB:
    def __init__(self,name, lvp_list=None, mode='m', path=None,load=True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sc = SimilarityCalculator()
        self.name = name
        self.mode = mode
        self.total_class = 0
        self.lvb_size = 0
        if lvp_list != None: self.generate_lvb(lvp_list,mode)
        elif load: self.load(path)
        print(f'{self.name} LVB: class {self.total_class} size {self.lvb_size}')
    
    def generate_confidence(self):
        max_values, max_indices = torch.min(self.var_bank, dim=1, keepdim=True)
        self.lv_confidence = torch.ones(self.lv_bank.shape[0],self.lv_bank.shape[1])
        #self.lv_confidence = 1 - self.var_bank/max_values
        #self.lv_bank *= self.lv_confidence
    
    def generate_lvb(self,lvp_list,mode='m'):
        lv_list = []
        var_list = []
        id_list = []
        self.total_class = len(lvp_list)
        for lvp in lvp_list:
            lv,var,id =  lvp.get_lvp(mode)
            lv_list.append(lv)
            var_list.append(var)
            id_list.append(id)
        self.lv_bank = torch.cat((lv_list))
        self.var_bank = torch.cat((var_list))
        self.id_bank = torch.cat((id_list))
        self.lvb_size = self.lv_bank.shape
    
    def eval_dataset(self,dataloader, dis_func='L1',with_confidence=False):
        acc,num_data = 0,0
        self.id_bank = self.id_bank.to(self.device)
        if not with_confidence:
            for image_features, labels in tqdm(dataloader):
                values, indices, similarity, similarity_raw = self.sc(self.lv_bank,image_features,dis_func=dis_func)
                indices = self.id_bank[indices]
                indices = torch.flatten(indices)
                result = torch.eq(labels.to(self.device),indices.to(self.device))
                acc += torch.sum(result)
                num_data += len(labels)
            
            
        else:
            self.generate_confidence()
            for image_features, labels in tqdm(dataloader):
                image_features = image_features.unsqueeze(1)
                image_features = image_features.repeat(1,len(self.lv_bank),1)
                #image_features = image_features * self.lv_confidence
                distence = torch.abs(image_features-self.lv_bank)
                
                var = self.var_bank.unsqueeze(0)
                var = var.repeat(len(image_features),1,1)
                #distence[distence > var] = distence[distence > var] / var[distence > var]

                distence = torch.sum(distence,dim=2) #/ torch.count_nonzero(distence,dim=2)
                
                indices = torch.argmin(distence, dim=1)
                indices = self.id_bank[indices]
                result = torch.eq(labels.to(self.device),indices.to(self.device))
                acc += torch.sum(result)
                num_data += len(labels)
        acc = acc.to(torch.float) / num_data
        print('accuracy: ',acc)
        return acc,num_data

    def extend(self,lv,id):
        self.lv_bank = torch.cat((self.lv_bank,lv))
        self.id_bank = torch.cat((self.id_bank,id))
        self.lvb_size = self.lv_bank.shape
    
    def save(self,path):
        data = {'lv':self.lv_bank,'var':self.var_bank,'id':self.id_bank, 'classes': self.total_class, 'size': self.lvb_size}
        save_tensor(data,path + f'{self.name}_{self.mode}.pt')
    
    def load(self,path): 
        data = torch.load(path + f'{self.name}_{self.mode}.pt')
        self.lv_bank = data['lv']
        self.var_bank = data['var']
        self.id_bank = data['id']
        self.total_class = data['classes']
        self.lvb_size = data['size']

class LVP:
    def __init__(self,id,lvp,pool_size = 10,cluster_size=10,dis_func='L1') -> None:
        self.lvp = lvp
        self.lvp_mean = torch.unsqueeze(torch.mean(self.lvp,dim=0),dim=0)
        self.lvp_var = torch.unsqueeze(torch.var(self.lvp,dim=0),dim=0)
        #self.lvp_confidence = 1 - self.lvp_var/torch.max(self.lvp_var)
        self.max_iterations = 50
        self.pool_size = pool_size
        self.cluster_size = cluster_size
        
        
        self.id = id
        self.sc = SimilarityCalculator()
        self.dis_func = dis_func
        print(f'LVP: {id}')
        print(f'pool size: {self.lvp.shape}')
    
    def get_lvp(self,mode='mean'):
        if mode == 'mean' or mode == 'm': 
            lvp = self.lvp_mean
            lvp_var = self.lvp_var
            lvp_id = torch.tensor([self.id],dtype=torch.long)
        elif mode == 'bank' or mode == 'b': 
            lvp = self.lvp
            lvp_var = None
            lvp_id = torch.tensor([self.id]*len(self.lvp),dtype=torch.long)
        elif mode == 'cluster' or mode == 'c': 
            lvp = self.lvp_clusters
            lvp_var = self.lvp_clusters_var
            lvp_id = torch.tensor([self.id]*len(self.lvp_clusters),dtype=torch.long)
        else:
            raise Exception("mode is not exist!")
        return lvp, lvp_var, lvp_id
        
    
    def extend(self,new_lvps):
        self.lvp = torch.cat((self.lvp,new_lvps))
        if len(self.lvp) > self.pool_size:
            self.generate_k_cluster(self.pool_size)
            self.lvp = self.lvp_clusters
        self.lvp_mean = torch.unsqueeze(torch.mean(self.lvp,dim=0),dim=0)
        self.lvp_var = torch.unsqueeze(torch.var(self.lvp,dim=0),dim=0)
        #self.lvp_confidence = 1 - self.lvp_var/torch.max(self.lvp_var)
    
    def extend_threshold(self,new_lvps,threshhold):
        self.lvp = torch.cat((self.lvp,new_lvps))
        self.generate_cluster_threshhold(threshhold)
        self.lvp_mean = torch.unsqueeze(torch.mean(self.lvp,dim=0),dim=0)
        self.lvp_var = torch.unsqueeze(torch.var(self.lvp,dim=0),dim=0)
    
    def generate_k_cluster(self,k=1,print_info=False):
        centroids,labels = k_means(self.lvp,k,dis_func=self.dis_func,max_iterations=self.max_iterations,print_info=print_info)
        self.lvp_clusters = centroids
        self.lvp_clusters_var = []
        for i in range(len(centroids)): self.lvp_clusters_var.append(torch.unsqueeze(torch.var(self.lvp[labels == i], dim=0),dim=0))
        self.lvp_clusters_var = torch.cat((self.lvp_clusters_var))
    
    def generate_cluster_threshhold(self,threshhold,print_info=True):
        centroids = self.lvp_mean
        self.lvp_clusters = centroids
        self.lvp_clusters_var = self.lvp_var
        
        while True:
            if len(centroids) == len(self.lvp):break
            values, indices, similarity, similarity_raw = self.sc(centroids,self.lvp,dis_func=self.dis_func)
            
            labels = indices.flatten().to('cpu')
            similarity_raw = similarity_raw.flatten().to('cpu')
            min_sim = torch.min(similarity_raw)
            
            if min_sim < threshhold:
                new_centroid = self.lvp[similarity_raw < threshhold]
                centroids = torch.cat((centroids,new_centroid))
                
                centroids = self.cluster_lv(centroids,self.lvp,print_info)
            else: break