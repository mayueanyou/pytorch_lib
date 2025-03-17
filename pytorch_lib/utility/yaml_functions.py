import torch,yaml
import numpy as np

def save_as_yaml(data,path):
    with open(path,'w') as f: yaml.dump(data,f)
    print(f'save as yaml: {path}')

def load_yaml(path,is_dataset=True):
    with open(path) as f: data = yaml.load(f,Loader=yaml.Loader)
    if is_dataset: 
        data['data'] = np.array(data['data'])
        data['targets'] = torch.tensor(data['targets']).to(dtype=torch.long)
    print(f'load yaml: {path}')
    return data