import os,sys,torch,cv2
import torchvision
from PIL import Image
from tqdm import tqdm
import numpy as np


def read_image(path,mode='PIL'):
    if mode == 'PIL':
        return torch.tensor(np.array(Image.open(path))).permute(2,0,1)
    elif mode == 'torch':
        return torchvision.io.read_image(path)
    elif mode == 'cv2':
        return torch.tensor(cv2.imread(path)).permute(2,0,1)

def save_rgb_image(image, path, mode = 'PIL'):
    if mode == 'PIL':
        if image.shape[-1] != 3: image = image.permute(1, 2, 0)
        image = image.to("cpu",torch.uint8).numpy()
        image = Image.fromarray(image)
        image.save(path)
    elif mode == 'torch':
        torchvision.utils.save_image(image, path)
    elif mode == 'cv2':
        image = image.permute(1, 2, 0).to("cpu",torch.uint8).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, image)
    print(f'Saved: {path}')


def save_dataset_images(dataset, save_path, dataset_name, transform=None):
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    for id in tqdm(range(torch.max(dataset.targets)+1)):
        if not os.path.exists(save_path + f'/{dataset_name}_{id}'): os.makedirs(save_path + f'/{dataset_name}_{id}')
        data = dataset.data[dataset.targets == id]
        for i in tqdm(range(len(data))): 
            save_rgb_image(data[i], save_path + f'/{dataset_name}_{id}/{i}.png')


