import os,sys,torch,yaml
import numpy as np

def save_as_yaml(data,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f: yaml.dump(data,f)
    print(f'save as yaml: {path}')

def load_yaml(path):
    with open(path) as f: data = yaml.load(f,Loader=yaml.Loader)
    print(f'load yaml: {path}')
    return data