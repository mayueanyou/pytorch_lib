import os,sys,torch,warnings,pathlib,yaml,torchvision
from .similarity_calculator import SimilarityCalculator
from tqdm import tqdm

def print_GPU_info():
    print('GPU Name: ',torch.cuda.get_device_name(0))  if torch.cuda.is_available() else print('No GPU')
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        print(f"Total GPU memory: {total_memory / (1024**2):.2f} MB")

def L1_similarity(label_features,input_features):
    input_features = input_features.unsqueeze(1)
    input_features = input_features.repeat(1,label_features.size(0),1)
    similarity = torch.abs(input_features - label_features)
    similarity = torch.sum(similarity,-1)
    return similarity
    
def L2_similarity(label_features,input_features):
    input_features = input_features.unsqueeze(1)
    input_features = input_features.repeat(1,label_features.size(0),1)
    similarity = torch.pow(input_features - label_features,2)
    similarity = torch.sum(similarity,-1)
    return similarity


def save_tensor(data,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data,path)
    print(f'Saved: {path}')


def k_means(data_pool,centroids,dis_func='L1',max_iterations=50,min_cluster_size = 5,print_info=False,batch_mode=False):
    sim_cal = SimilarityCalculator(batch_mode=batch_mode)
    if type(centroids) == int: 
        if centroids > len(data_pool): 
            warnings.warn("centroids should be less than the length of data_pool")
            return data_pool,None
        else: centroids = data_pool[torch.randperm(len(data_pool))[:centroids]]
    
    values, indices, similarity, distance = sim_cal(centroids,data_pool,dis_func = dis_func)
    labels_current = indices.flatten().to('cpu')
    
    #for step in range(max_iterations):
    for step in tqdm(range(max_iterations)):
        for i in range(len(centroids)):
            sys.stdout.flush()
            if torch.sum(labels_current == i) < min_cluster_size: 
                centroids[i] = data_pool[torch.randperm(len(data_pool))[0]]
                #centroids[i] += torch.randn(centroids[i].shape) * torch.mean(centroids[i]) * 0.01
            else: 
                centroids[i] = torch.mean(data_pool[labels_current == i], dim=0)
                
            if print_info: print(f'{i}, {len(labels_current[labels_current==i])}', end=' | ')
        if print_info: print()
        
        values, indices, similarity, similarity_raw = sim_cal(centroids,data_pool,dis_func=dis_func)
        labels_new = indices.flatten().to('cpu')
        
        if torch.equal(labels_current,labels_new): break
        labels_current = labels_new
    
    return centroids,labels_current
