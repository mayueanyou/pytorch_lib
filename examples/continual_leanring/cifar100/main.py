import os,sys,torch,random,argparse
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Compose

import pytorch_lib as ptl

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

dataset_path = '/home/yma183/datasets/CIFAR100'
current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def generate_experiment(args):
    train_data= ptl.dataset_lib.dataset.scan_image_folder(dataset_path+'/train')
    train_data = ptl.dataset_lib.dataset.CustomDataset(train_data['data'],train_data['targets'])
    test_data= ptl.dataset_lib.dataset.scan_image_folder(dataset_path+'/test')
    test_data = ptl.dataset_lib.dataset.CustomDataset(test_data['data'],test_data['targets'])
    tasks_num = [5,10,25,50,100]
    for tasks in tasks_num:
        unit = 100//tasks
        tasks_id = 1
        for i in range(0,100,unit):
            targets_select = [j for j in range(i,i+unit)]
            #print(targets_select)
            dataset = ptl.select_by_label(train_data,targets_select)
            ptl.save_as_yaml({'data':dataset.data.tolist(),'targets':dataset.targets.tolist() }, current_path + f'/experiments/{tasks}_tasks/train/{tasks_id}.yaml')
            tasks_id += 1
        
    #test = ptl.select_by_label(train_data,[0,1,2,3])
    #ptl.save_as_yaml({'data':test.data.tolist(),'targets':test.targets.tolist() }, current_path + '/experiments/10_tasks/test.yaml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="None")
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args)