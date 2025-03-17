import os,sys,torch
import torchvision
import matplotlib.pyplot as plt

def img_to_patch(img,patch_size):
    B, C, H, W = img.shape
    img = img.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    img = img.permute(0,2,4,1,3,5)#[B,H,W,C,ph,pw]
    img = img.flatten(1,2)#[B,H*W,C,ph,pw]
    return img

def plot_patches(img,image_size,patch_size):
    img_patches = img_to_patch(img,patch_size)
    fig,ax = plt.subplots(img.shape[0],1)
    for i in range(img.shape[0]):
        img_grid = torchvision.utils.make_grid(img_patches[i],nrow = image_size//patch_size, normalize=True, pad_value= 0.9)
        img_grid = img_grid.permute(1,2,0)
        ax[i].imshow(img_grid)
        ax[i].axis('off')
    plt.savefig('./test.png')

if __name__ == '__main__':
    data = torch.rand(2,1,28,28)
    
    plot_patches(data,28,7)
    pass