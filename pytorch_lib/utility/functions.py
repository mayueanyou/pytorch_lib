import os,sys,torch,warnings,pathlib,yaml,torchvision
import numpy as np
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


def save_data(path,data,mode='tensor'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if type(data) == np.ndarray: np.save(path,data)
    elif type(data) == torch.Tensor or mode=='tensor': torch.save(data,path)
    else: raise ValueError("Unsupported data type. Use numpy array or torch tensor")
    print(f'Saved: {path}')

def load_data(path):
    if os.path.isfile(path):
        if path.endswith('.npy'): data = np.load(path)
        elif path.endswith('.pt'): data = torch.load(path)
        else: raise ValueError("Unsupported file format. Use .npy or .pt")
    else: raise FileNotFoundError(f"File not found: {path}")
    print(f'Loaded: {path}')
    return data


def rate_from_tensors(test_data,candidates,data_pool,topk=1):
    sc = SimilarityCalculator()
    candidate_data = data_pool[candidates].view(-1,data_pool.shape[-1])
    candidate_targets = candidates.repeat_interleave(data_pool.shape[1])
    
    values, indices, similarity, similarity_raw = sc(candidate_data,test_data,topk = topk)
    indices = indices.flatten()
    score = similarity_raw.flatten()
    
    score = score[indices]
    indices = candidate_targets[indices]
    
    candidate_count = torch.tensor([(indices == it).sum().item() for it in candidates])
    candidate_score = torch.tensor([(score[indices == it]).sum().item() for it in candidates])
    
    #predict = candidates[torch.argmax(torch.tensor(candidate_score))]
    predict = candidates[torch.argmax(torch.tensor(candidate_count))]
    
    return predict, candidate_score, candidate_count

def count_max_items(tensor):
    unique, counts = torch.unique(tensor, return_counts=True)
    max_count_index = torch.argmax(counts)
    max_item = unique[max_count_index]
    max_count = counts[max_count_index]
    return max_item, max_count
