import os,sys,torch,warnings,pathlib,yaml,torchvision
from . import*

def save_tensor(data,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data,path)
    print(f'Saved: {path}')



def scan_image_folder(path):
    path = pathlib.Path(path)
    folders = [item for item in sorted(path.iterdir(), key=lambda x: x.name)]
    dataset = {'data':[],'targets':[]}
    class_id = 0
    for folder in folders:
        current_images = [item.as_posix() for item in folder.iterdir()]
        dataset['data'] += current_images
        dataset['targets'] += [class_id]*len(current_images)
        class_id += 1
    return dataset


def k_means(data_pool,centroids,dis_func='L1',max_iterations=50,min_cluster_size = 5,print_info=False):
    sim_cal = SimilarityCalculator()
    if type(centroids) == int: 
        if centroids > len(data_pool): 
            warnings.warn("centroids should be less than the length of data_pool")
            return data_pool,None
        else: centroids = data_pool[torch.randperm(len(data_pool))[:centroids]]
    
    values, indices, similarity, distance = sim_cal(centroids,data_pool,dis_func = dis_func)
    labels_current = indices.flatten().to('cpu')
    
    for step in range(max_iterations):
        for i in range(len(centroids)):
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
        else: labels_current = labels_new
    
    return centroids,labels_current
