import os,cv2,sys,torch,random,argparse,torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

torch.set_printoptions(precision=2, threshold=100000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)
np.set_printoptions(precision=1, threshold=100000, linewidth=10000)

dataset_path = "~/datasets"

def main():
    def imshow(img,id,target):
        print(img.shape)
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        img_gray = torchvision.transforms.functional.rgb_to_grayscale(img)
        img_gray = img_gray.numpy()[0]
        img = img.numpy()
        
        #cv2.imwrite('./images/%d-%d-gray.png'%(id,target), img_gray)
        #cv2.imwrite('./images/%d-%d-red.png'%(id,target), img[0])
        #cv2.imwrite('./images/%d-%d-green.png'%(id,target), img[1])
        #cv2.imwrite('./images/%d-%d-blue.png'%(id,target), img[2])
        data =  np.nan_to_num(img[2]/img[1])
        data /= np.max(data)
        data *= 255
        print(data)
        cv2.imwrite('./images/%d-%d-gb.png'%(id,target),data)
        #plt.imshow(np.transpose(img, (0, 1, 2)))
        #plt.savefig('./images/%d-%d.png'%(id,target))
        #plt.show()
    training_data = datasets.CIFAR10(root=dataset_path,train=True,download=True,transform=ToTensor())
    for i in range(10):
        imshow(training_data.data[i],i,training_data.targets[i])
        break
    

if __name__ == "__main__":
    main()