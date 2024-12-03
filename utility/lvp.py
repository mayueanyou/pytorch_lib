import os,sys,torch,copy
from tqdm import tqdm
from . import*

class LVB:
    def __init__(self,name, lvp_list=None, mode='m', path=None,load=True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sc = SimilarityCalculator()
        self.name = name
        self.mode = mode
        if lvp_list != None: self.generate_lvb(lvp_list,mode)
        elif load: self.load(path)
        print(f'LVB: class {self.total_class} size {self.lvb_size}')
    
    def generate_lvb(self,lvp_list,mode='m'):
        lv_list = []
        id_list = []
        self.total_class = len(lvp_list)
        for lvp in lvp_list:
            lv,id =  lvp.get_lvp(mode)
            lv_list.append(lv)
            id_list.append(id)
        self.lv_bank = torch.cat((lv_list))
        self.id_bank = torch.cat((id_list))
        self.lvb_size = self.lv_bank.shape
    
    def eval_dataset(self,dataset, dis_func='L1'):
        acc,num_data = 0,0
        self.id_bank = self.id_bank.to(self.device)
        for image_features, labels in tqdm(dataset):
            values, indices, similarity = self.sc(self.lv_bank,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            indices = self.id_bank[indices]
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            num_data += len(labels)
        acc = acc.to(torch.float) / num_data
        print('accuracy: ',acc)
        return acc,num_data
    
    def save(self,path): 
        torch.save({'lv':self.lv_bank,'id':self.id_bank, 'classes': self.total_class, 'size': self.lvb_size},path + f'{self.name}_{self.mode}.pt')
    
    def load(self,path): 
        data = torch.load(path + f'{self.name}_{self.mode}.pt')
        self.lv_bank = data['lv']
        self.id_bank = data['id']
        self.total_class = data['classes']
        self.lvb_size = data['size']

class LVP:
    def __init__(self,id,lvp_bank,dis_func='L1') -> None:
        self.lvp_bank = lvp_bank
        self.lvp_mean = torch.unsqueeze(torch.mean(self.lvp_bank,dim=0),dim=0)
        self.id = id
        self.sc = SimilarityCalculator()
        self.dis_func = dis_func
        print(f'LVP: {id}')
        print(f'bank shape: {self.lvp_bank.shape}')
    
    def get_lvp(self,mode='mean'):
        if mode == 'mean' or mode == 'm': return self.lvp_mean, torch.tensor([self.id],dtype=torch.long)
        elif mode == 'bank' or mode == 'b': return self.lvp_bank, torch.tensor([self.id]*len(self.lvp_bank),dtype=torch.long)
        elif mode == 'cluster' or mode == 'c': return self.lvp_clusters, torch.tensor([self.id]*len(self.lvp_clusters),dtype=torch.long)
        print("mode is not exist!")
    
    def extend(self,new_lvps):
        self.lvp_bank = torch.cat((self.lvp_bank,new_lvps))
        self.lvp_mean = torch.unsqueeze(torch.mean(self.lvp_bank,dim=0),dim=0)
    
    def extend_threshold(self,new_lvps,threshold):
        values, indices, similarity = self.sc(self.lvp_mean[None,:],new_lvps,dis_func=self.dis_func)
        select_lvps = new_lvps[similarity >= threshold]
        rest_lvps = new_lvps[similarity <threshold]
        if len(select_lvps) == 0: return False,None
        self.extend(select_lvps)
        return True,rest_lvps
    
    def generate_k_cluster(self,k=1,iterations=10,print_info=True):
        if type(k) is int: centroids = self.lvp_bank[torch.randperm(self.lvp_bank.size(0))[:k]]
        else: 
            centroids = copy.deepcopy(k)
            k = len(centroids)
        for _ in range(iterations):
            values, indices, similarity = self.sc(centroids,self.lvp_bank,dis_func=self.dis_func)
            _, labels = torch.max(similarity, dim=1)
            labels = labels.to('cpu')

            for i in range(k):
                if torch.sum(labels == i) > 0:
                    if print_info: print(f'{i}, {len(labels[labels==i])}', end=' | ')
                    centroids[i] = torch.mean(self.lvp_bank[labels == i], dim=0)
            if print_info: print()
        #print(centroids.shape)
        self.lvp_clusters = centroids
        label = torch.ones(k) * self.id
        label = label.to(torch.long)
        return self.lvp_clusters, label