import os,sys,copy,torch,random,cv2,torchvision,skimage
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF
import numpy as np
from PIL import Image
from abc import ABC,abstractmethod

class Transform:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def __call__(self) -> None:pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class COCOTargetsTF(Transform):
    def __init__(self,height,width) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def process_segmentation(self,segmentation, width, height):
        if isinstance(segmentation, list):
            from skimage.draw import polygon
            polygon_list = segmentation
            mask_list = []
            if len(polygon_list) == 0: 
                mask_list.append(torch.zeros((self.height, self.width), dtype=torch.bool))
                mask_list = torch.stack(mask_list, dim=0)
            else:
                for j in range(len(polygon_list)):
                    polygon_item = polygon_list[j]
                    polygon_item = torch.tensor(polygon_item).reshape(-1, 2)
                    mask_np = np.zeros((height, width), dtype=np.uint8)
                    rr, cc = polygon(polygon_item[:, 1], polygon_item[:, 0], shape=(height, width))
                    mask_np[rr, cc] = 1
                    mask_np = skimage.transform.resize(mask_np, (self.height, self.width), anti_aliasing=False)
                    mask = torch.from_numpy(mask_np).to(torch.bool)
                    mask_list.append(mask)
                if len(mask_list) > 1:
                    mask = mask_list[0]
                    for i in range(1, len(mask_list)):
                        mask = torch.logical_or(mask, mask_list[i])
                else:
                    mask = mask_list[0]

        else:
            import pycocotools.mask as mask_util
            rle_dict = segmentation
            rle_dict = mask_util.frPyObjects(rle_dict, rle_dict['size'][0], rle_dict['size'][1])
            mask_np = mask_util.decode(rle_dict)
            mask = skimage.transform.resize(mask_np, (self.height, self.width), anti_aliasing=False)
            mask = torch.from_numpy(mask).to(torch.bool)
        return mask
    
    def __call__(self, x):
        boxes =[]
        labels = []
        masks = []
        for item in x:
            boxes.append(torch.tensor(item['boxes'],dtype=torch.float32))
            labels.append(torch.tensor(item['labels'],dtype=torch.int64))
            masks.append(self.process_segmentation(item['segmentation'], item['width'], item['height']))
        targets = {
            'boxes': torch.stack(boxes, dim=0),
            'labels': torch.stack(labels, dim=0),
            'masks': torch.stack(masks, dim=0),
            'width': self.width,
            'height': self.height,
        }
        return targets

class FlattenTF(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x):
        return torch.flatten(x)

class ReadImageTF(Transform):
    def __init__(self,mode='PIL') -> None:
        super().__init__()
        self.mode = mode
    
    def __call__(self, path):
        if self.mode == 'PIL':
            #print(f'Read Image: {path}')
            image = Image.open(path)
            return image.convert("RGB")
            
        elif self.mode == 'torch':
            return torchvision.io.read_image(path, mode = 'RGB').to(torch.float32)
            #return torchvision.io.decode_image(input=path,mode = 'RGB')

class ImgUpscaleTF(Transform):
    def __init__(self,times=2) -> None:
        super().__init__()
        self.times = times

class ImgRepeatTF(Transform):
    def __init__(self,times=2) -> None:
        super().__init__()
        self.times = times
    
    def repeat_image(self, image, times_x, times_y):
        """Repeats an image horizontally and vertically."""

        width, height = image.size
        new_width = width * times_x
        new_height = height * times_y

        new_image = Image.new('RGB', (new_width, new_height))

        for x in range(times_x):
            for y in range(times_y):
                new_image.paste(image, (x * width, y * height))

        return new_image
    
    def __call__(self, x):
        x = self.repeat_image(x, self.times, self.times)
        return x

class SoftmaxTF(Transform):
    def __init__(self,dim) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        x = x.softmax(self.dim)
        return x

class ScaleTF(Transform):
    def __init__(self,scale_tensor) -> None:
        super().__init__()
        self.scale_tensor = scale_tensor

    def __call__(self, x):
        x *= self.scale_tensor
        return x

class RandMixTF(Transform):
    def __init__(self,file_path,ex_rate=1,softmax=True) -> None:
        super().__init__()
        self.mix_tensor = torch.load(file_path)
        self.ex_rate = ex_rate
        self.softmax = softmax

    def __call__(self, x):
        extend_feature = x @ self.mix_tensor.T
        if self.softmax: extend_feature = extend_feature.softmax(-1)
        extend_feature *= self.ex_rate
        x = torch.cat((x,extend_feature))
        return x

class Slice(Transform):
    def __init__(self,idx_range) -> None:
        super().__init__()
        self.idx_range = idx_range

    def __call__(self, x):
        x = x[..., self.idx_range[0]:self.idx_range[1]]
        return x


class To_Tensor_Noise(Transform):
    def __init__(self,shape) -> None:
        super().__init__()
        self.shape = shape

    def __call__(self, pic):
        pic = TF.to_tensor(pic)
        pic += torch.randn(self.shape[0], self.shape[1], self.shape[2])*0.1
        return pic

class RGB_Add_Gray(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        pic = TF.to_tensor(pic)
        gray_pic = TF.rgb_to_grayscale(pic)
        return torch.cat((pic,gray_pic))


class PermuteChannle(Transform):
    def __init__(self,order) -> None:
        super().__init__()
        self.order = order

    def __call__(self, data):
        data = data.permute(self.order)
        return data


class PermuteColor(Transform):
    def __init__(self,order) -> None:
        super().__init__()
        self.order = order

    def __call__(self, data):
        data = data[self.order,:]
        return data

class RGB_Extension(Transform):
    def __init__(self) -> None:
        super().__init__()

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


class Gray_Add_Color(Transform):
    def __init__(self) -> None:
        super().__init__()
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
