import os,sys,copy,torch,random,cv2,torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF

class To_Tensor_Noise:
    def __init__(self,shape) -> None:
        self.shape = shape

    def __call__(self, pic):
        pic = TF.to_tensor(pic)
        pic += torch.randn(self.shape[0], self.shape[1], self.shape[2])*0.1
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class RGB_Add_Gray:
    def __init__(self) -> None:
        pass

    def __call__(self, pic):
        pic = TF.to_tensor(pic)
        gray_pic = TF.rgb_to_grayscale(pic)
        return torch.cat((pic,gray_pic))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class RGB_Extension:
    def __init__(self) -> None:
        pass

    def __call__(self, pic):
        self.pic = TF.to_tensor(pic)
        new_pic = self.add_color(self.pic,0,0.5,0.5)
        new_pic = self.add_color(new_pic,0.5,0.5,0)
        new_pic = self.add_color(new_pic,0.5,0,0.5)
        return new_pic
    
    def add_color(self,new_pic,r,g,b):
        weights = torch.tensor([[r],[g],[b]],dtype=torch.float).view(3, 1, 1, 1)
        new_color = torch.sum(NNF.conv2d(self.pic, weights,groups=3),dim=0)[None,:]
        return torch.cat((new_pic,new_color),dim=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Gray_Add_Color:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        idx_list = []
        self.color_range = 3
        for i in range(self.color_range):
            for j in range(self.color_range):
                for k in range(self.color_range):
                   idx_list.append([[i,j,k]]) 
        self.idx_tensor = torch.tensor(idx_list).to(self.device)

    def __call__(self, pic):
        pic = TF.to_tensor(pic).to(self.device)
        pic_t = pic.permute(*torch.arange(pic.ndim - 1, -1, -1))
        colors = self.get_colors(pic_t.reshape(-1,3))
        gray_pic = TF.rgb_to_grayscale(pic)
        gray_pic = gray_pic.view(-1)
        return torch.cat((gray_pic,colors))
    
    def get_colors(self,pic):
        colors = torch.zeros((self.color_range,self.color_range,self.color_range)).to(self.device)
        pic = torch.round(pic,decimals=1)*1.9
        pic = pic.to(torch.int)
        pic = torch.stack([pic for i in range(27)])
        idx = (pic == self.idx_tensor)
        idx = torch.count_nonzero(idx,dim=2)
        idx = torch.count_nonzero(idx==3,dim=1)
        colors = idx/idx.sum()
        return colors

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"