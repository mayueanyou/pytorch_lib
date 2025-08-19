import os,sys,torch,random,argparse
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms,tv_tensors
from torchvision.transforms import ToTensor,v2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import pytorch_lib as ptl
import python_lib as pyl

torch.set_printoptions(precision=2, threshold=1000000, edgeitems=None, linewidth=1000000, profile=None, sci_mode=False)
np.set_printoptions(precision=4, threshold=10000000, edgeitems=None, linewidth=10000000, suppress=True)

current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def main():
    transforms = v2.Compose(
    [
        ptl.ReadImageTF(),
        v2.ToImage(),
        v2.Resize(size=(480, 640)),
        # v2.RandomPhotometricDistort(p=1),
        # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        # v2.RandomIoUCrop(),
        # v2.RandomHorizontalFlip(p=1),
        # v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
    coco = ptl.COCOCustom('/datasets/COCO',training_transform=transforms, test_transform=transforms, image_size=(480, 640))
    train_datasetloader, test_datasetloader = coco.get_loaders(batch_size=1)
    ploter = pyl.Ploter()
    
    for imgs, targets in tqdm(train_datasetloader):
        imgs = list(imgs)
        targets = list(targets)
        data = {'image': torch.tensor(imgs[0]).permute(1, 2, 0).numpy(), 'boxes': targets[0]['boxes'].numpy(), 'labels': targets[0]['labels'].numpy(), 'masks': targets[0]['masks'].numpy(),
                'width': targets[0]['width'], 'height': targets[0]['height'],'plot_boxes': True, 'plot_masks': True}
        ploter.plot_detection(current_path + '/test.png', data)
        input()
    
    return 

    
    net = ptl.TorchModel('maskrcnn_resnet50_fpn_v2').net.to(ptl.device)
    #net.eval()
    net.train()
    
    #print(len(train_datasetloader))
    #print(len(test_datasetloader))
    #input()

    # sample = train_dataset[0]
    # img, target = sample
    # #print(f"{type(img) = }\n{type(target) = }\n{type(target[0]) = }\n{target[0].keys() = }")
    # print(img.shape)
    # print(len(target))
    # print(target.keys())
    # print(target['boxes'].shape)
    # print(target['labels'].shape)
    # print(target['masks'].shape)
    # print(target['masks'][0])
    count = 0
    no_list = []
    ploter = pyl.Ploter()

    for imgs, targets in tqdm(test_datasetloader):
        imgs = list(imgs)
        targets = list(targets)
        #data = {'image': torch.tensor(imgs[0]).permute(1, 2, 0).numpy(), 'boxes': targets[0]['boxes'].numpy(), 'labels': targets[0]['labels'].numpy(), 'masks': targets[0]['masks'].numpy()}
        #ploter.plot_detection(current_path + '/test.png', data)
        #return
        #input()
        if 'boxes' not in targets[0].keys():
            count += 1
            no_list.append(targets[0]['image_id'])
            #print(targets[0])
            #input()
            continue
            
        
        
        # #print(targets[0].keys())
        # for i in range(len(imgs)):
        #     imgs[i] = imgs[i].to(ptl.device)
        #     targets[i] = {k: torch.tensor(v).to(ptl.device) for k, v in targets[i].items()}
        # print(targets[0]['boxes'].shape)
        # #pre = net(imgs,targets)
        # #print(pre)
        # input()
    print(f"Total samples without boxes: {count}")
    print(no_list)



if __name__ == "__main__":
    main()