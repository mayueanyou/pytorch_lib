import os,sys,copy,torch,pathlib,yaml
import numpy as np

def select_by_label(dataset,target_list):
    dataset = copy.deepcopy(dataset)
    dataset.targets = torch.tensor(dataset.targets)
    idx = sum(dataset.targets==i for i in target_list).bool()
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    print(f'select label:\n {target_list}')
    return dataset

def change_label(dataset,label_setup):
    dataset = copy.deepcopy(dataset)
    dataset.targets = torch.tensor(dataset.targets)
    for setup in label_setup:
        idx = sum(dataset.targets==i for i in setup[0]).bool()
        dataset.targets[idx] = setup[1]
    print('change label:')
    for it in label_setup: print(f'{it[0]} -> {it[1]}')
    return dataset

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
    dataset['data'] = np.array(dataset['data'])
    dataset['targets'] = np.array(dataset['targets'])
    return dataset

def save_dataset_as_yaml(dataset,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {'data':dataset['data'].tolist(),'targets':dataset['targets'].tolist()}
    with open(path,'w') as f: yaml.dump(data,f)
    print(f'save dataset as yaml: {path}')

def load_dataset_from_yaml(path):
    with open(path) as f: data = yaml.load(f,Loader=yaml.Loader)
    data['data'] = np.array(data['data'])
    data['targets'] = np.array(data['targets']) #torch.tensor(data['targets']).to(dtype=torch.long)
    print(f'load dataset from yaml: {path}')
    return data