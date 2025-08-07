import os,sys,torch
from tqdm import tqdm
from .similarity_calculator import SimilarityCalculator

def k_means(data_pool,centroids,dis_func='L1',max_iterations=100,min_cluster_size = 4,batch_mode=False,repick=True):
    sim_cal = SimilarityCalculator()
    
    def random_generate_k_centroids(centroids):
        if type(centroids) == int: 
            if centroids > len(data_pool): 
                warnings.warn("centroids should be less than the length of data_pool")
                return data_pool,None
            else: 
                centroids = data_pool[torch.randperm(len(data_pool))[:centroids]]
                #centroids = torch.randn(centroids.shape) * torch.mean(data_pool) + torch.mean(data_pool,dim=0,keepdim = True)
        return centroids

    centroids = random_generate_k_centroids(centroids)
    
    average_group = len(data_pool) / len(centroids)
    if average_group < min_cluster_size: min_cluster_size = 2 
    
    indices= sim_cal.only_get_indices(centroids,data_pool,dis_func = dis_func,batch_mode=batch_mode)
    labels_current = indices.flatten().to('cpu')
    
    for step in range(max_iterations):
    #for step in tqdm(range(max_iterations)):
        for i in range(len(centroids)):
            
            if torch.sum(labels_current == i) < min_cluster_size: 
                if repick: centroids[i] = data_pool[torch.randperm(len(data_pool))[0]]
                else: centroids[i] += torch.randn(centroids[i].shape) * torch.mean(centroids[i])
            else: 
                centroids[i] = torch.mean(data_pool[labels_current == i], dim=0)
        
        indices = sim_cal.only_get_indices(centroids,data_pool,dis_func=dis_func,batch_mode=batch_mode)
        labels_new = indices.flatten().to('cpu')
        
        if torch.equal(labels_current,labels_new): break
        labels_current = labels_new
    
    grout_status = torch.tensor([torch.sum(labels_current == i) for i in range(len(centroids))])
    print('k_means status:')
    print(f'steps: [{step}] dis_func: [{dis_func}]')
    print('group size: ',grout_status)
    
    #centroids = centroids[grout_status > min_cluster_size]
    
    return centroids,labels_current

class KmeansCalculator():
    def __init__(self,dis_func='L1',batch_mode=False,mode='from_mean',from_mean_rate=1):
        self.sc = SimilarityCalculator()
        self.group_dis_func = dis_func
        self.evalue_dis_func = 'Cos'
        self.batch_mode = batch_mode
        self.mode = mode
        #self.adjust_mode = 'from_mean'
        self.adjust_mode = 'min_cluster_size'
        self.resample_mode = 'from_mean'
        self.from_mean_rate = from_mean_rate
        if mode not in ['from_mean','from_pool']: raise Exception('KmeansCalculator: mode should be from_mean or from_pool')
        
        self.distance = {'among_centroids':0,
                        'centroids_between_mean':0,
                        'pool_between_mean':0,
                        'group_sum':0}


    def calculate_score(self,total_distance,distance_among_centroids,distance_centroids_between_mean):
        return total_distance * 0.0172283431462576 + distance_among_centroids * -22.2920377147066 + distance_centroids_between_mean * 35.248464661093
    
    def calculate_distance(self,label_features,input_features):
        _, _, _, similarity_raw = self.sc(label_features,input_features,dis_func=self.evalue_dis_func,batch_mode=self.batch_mode)
        if self.evalue_dis_func == 'Cos': similarity_raw = self.sc.convert_Cos_similarity_to_distance(similarity_raw)
        else: similarity_raw /= label_features.shape[-1]
        distance = torch.sum(similarity_raw)
        return distance

    def resample_centroid(self,data_pool,centroid):
        data_pool_mean = torch.mean(data_pool,dim=0,keepdim=True)
        if self.resample_mode == 'from_mean':
            #centroid = ((torch.randn(centroid.shape)*2-1) * self.from_mean_rate + 1) * data_pool_mean
            centroid = torch.normal(mean=data_pool_mean, std=self.from_mean_rate)
        elif self.resample_mode == 'from_pool':
            centroid = data_pool[torch.randperm(len(data_pool))[0]]
        else: raise Exception('KmeansCalculator: resample_mode should be from_mean or from_pool')
        return centroid

    def update_distance(self,data_pool,centroids,data_labels):
        self.distance['pool_between_mean'] = self.calculate_distance(torch.mean(data_pool,dim=0,keepdim=True),data_pool)
        self.distance['among_centroids'] = self.calculate_distance(centroids,centroids)/2/torch.sum(torch.arange(len(centroids)))
        self.distance['centroids_between_mean'] = self.calculate_distance(torch.mean(data_pool,dim=0,keepdim=True),centroids) / len(centroids)
        self.distance['group_sum'] = self.calculate_group_sum(data_pool,centroids,data_labels)

    def calculate_group_sum(self,data_pool,centroids,data_labels):
        total_distance = 0
        for i in range(len(centroids)):
            centroid =centroids[i]
            data_near_centroid = data_pool[data_labels == i]
            if len(data_near_centroid) == 0: continue
            distance = self.calculate_distance(centroid[None,:],data_near_centroid)
            total_distance += distance
        return total_distance
    
    def random_generate_k_centroids(self,data_pool,centroids):
        data_pool_mean = torch.mean(data_pool,dim=0,keepdim=True)
        if centroids >= len(data_pool): 
            raise Exception(f'data_pool size: {data_pool.shape}, kmeans: {centroids} \n centroids should be less than the length of data_pool')
        else: centroids = data_pool[torch.randperm(len(data_pool))[:centroids]]

        if self.mode == 'from_mean': #centroids = ((torch.randn(centroids.shape)*2-1) * self.from_mean_rate + 1) * data_pool_mean
            centroids = torch.normal(mean=data_pool_mean.repeat(centroids.shape[0], 1), std=self.from_mean_rate)
        return centroids

    def adjust_min_cluster_size(self,data_pool,centroids,min_cluster_size):
        average_group = len(data_pool) / len(centroids)
        if average_group < min_cluster_size: min_cluster_size = 2
        return min_cluster_size

    def adjust_centroid(self,data_pool,centroid,label_select,adjust_mode='from_mean'):
        data_pool_mean = torch.mean(data_pool,dim=0,keepdim=True)
        resample = False
        if adjust_mode == 'from_mean':
            normalized_distance_to_mean = torch.sum(torch.cdist(centroid, data_pool_mean,p=1))/torch.sum(torch.abs(data_pool_mean))
            if normalized_distance_to_mean > self.from_mean_rate: 
                centroid = self.resample_centroid(data_pool,centroid)
                resample = True
        
        elif adjust_mode == 'min_cluster_size':
            if torch.sum(label_select) < 4:
                centroid = self.resample_centroid(data_pool,centroid)
                resample = True
        
        elif adjust_mode == 'from_pool':
            if torch.sum(label_select) < 4:
                centroid = data_pool[torch.randperm(len(data_pool))[0]]
                resample = True
        
        if not resample: centroid = torch.mean(data_pool[label_select], dim=0)
        return centroid
    
    def remove_small_group(self,data_pool,centroids,labels_current,min_cluster_size,batch_mode):
        grout_status = torch.tensor([torch.sum(labels_current == i) for i in range(len(centroids))])
        centroids = centroids[grout_status > min_cluster_size]
        values, indices, similarity, similarity_raw = self.sc(centroids,data_pool,dis_func=self.group_dis_func,batch_mode=batch_mode)
        data_labels = indices.flatten().to('cpu')
        #self.print_info(data_pool,0,data_labels,centroids,self.group_dis_func,min_cluster_size)
        return centroids,data_labels

    def print_info(self,data_pool,step,labels_current,centroids,dis_func,min_cluster_size):
        grout_status = torch.tensor([torch.sum(labels_current == i) for i in range(len(centroids))])
        print('k_means status:')
        print(f'data_pool: [{data_pool.shape}] k: [{len(centroids)}] steps: [{step}] dis_func: [{dis_func}] min_cluster_size [{min_cluster_size}]')
        print('group size: ',grout_status)

    def __call__(self,data_pool,centroids,max_iterations=100,min_cluster_size = 4,batch_mode=False,adjust_mode='min_cluster_size',remove=False,verbose=True):
        if type(centroids) == int: 
            if centroids ==1: return torch.mean(data_pool,dim=0,keepdim=True), [[0]]
            else: centroids = self.random_generate_k_centroids(data_pool,centroids)
        min_cluster_size = self.adjust_min_cluster_size(data_pool,centroids,min_cluster_size)
        
        values, indices, similarity, distance = self.sc(centroids,data_pool,dis_func = self.group_dis_func,batch_mode=batch_mode)
        labels_current = indices.flatten().to('cpu')
        
        for step in range(max_iterations):
            for i in range(len(centroids)):
                label_i = labels_current == i
                centroid_i = centroids[i][None,:]
                centroids[i] = self.adjust_centroid(data_pool,centroid_i,label_i,adjust_mode=adjust_mode)
            
            values, indices, similarity, similarity_raw = self.sc(centroids,data_pool,dis_func=self.group_dis_func,batch_mode=batch_mode)
            labels_new = indices.flatten().to('cpu')
            
            if torch.equal(labels_current,labels_new): break
            labels_current = labels_new
        
        data_labels = labels_current
        
        if verbose: self.print_info(data_pool,step,data_labels,centroids,self.group_dis_func,min_cluster_size)
        if remove: centroids, data_labels = self.remove_small_group(data_pool,centroids,data_labels,min_cluster_size,batch_mode)
        return centroids,data_labels
    
    def iterative_generation(self,data_pool,k=10,k_range=[(2,20)],max_iterations=100,min_cluster_size = 4,batch_mode=False,remove=False):
        original_data_pool = data_pool.clone().detach()
        for range_setp in k_range:
            new_data_pool = []
            for k_step in range(range_setp[0],range_setp[1]+1):
                current_centroids, _ = self(data_pool,k_step,max_iterations=100,min_cluster_size = min_cluster_size,batch_mode=False,verbose=False)
                new_data_pool.append(current_centroids)
            new_data_pool = torch.cat((new_data_pool))
            data_pool = new_data_pool
        
        centroids,_ = self(data_pool,k,max_iterations=max_iterations,min_cluster_size=2,batch_mode=batch_mode,adjust_mode='from_pool')
        
        values, indices, similarity, similarity_raw = self.sc(centroids,original_data_pool,dis_func=self.group_dis_func,batch_mode=batch_mode)
        data_labels = indices.flatten().to('cpu')
        print(f'k_range: {k_range}')
        self.print_info(original_data_pool,0,data_labels,centroids,self.group_dis_func,min_cluster_size)
        if remove: centroids, data_labels = self.remove_small_group(original_data_pool,centroids,data_labels,min_cluster_size,batch_mode)
        return centroids,data_labels