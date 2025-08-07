import os,sys,torch,random,argparse
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lib as ptl


def main():
    coco = ptl.COCO('/home/yma183/datasets/COCO')
    net = ptl.TorchModel('maskrcnn_resnet50_fpn_v2').net
    
    sample = coco.training_data[0]
    img, target = sample
    print(f"{type(img) = }\n{type(target) = }\n{type(target[0]) = }\n{target[0].keys() = }")


if __name__ == "__main__":
    main()