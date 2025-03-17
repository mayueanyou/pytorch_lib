import os,sys,timm,torch
from pytorch_lib import*
#from python_lib import*
from tqdm import tqdm

class VitWrapper:
    def __init__(self,weight='vit_base_patch16_224') -> None:
        self.weight = weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = timm.create_model(weight,pretrained=True,num_classes=0)
        self.net.to(self.device)
        self.net.eval()
        data_config = timm.data.resolve_model_data_config(self.net)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        print("Vit: -", self.weight)
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f'total parameters: {total_params:,}')
        print('transfors: -',self.transforms)
        print('='*100)
    
    def inference(self,images):
        return self.net(images)
        return self.net(self.transforms(images))
    
    def inference_dataset(self,dataset):
        data  = {'data':[],'targets':[]}
        for images, labels in tqdm(dataset):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.inference(images)
            
            data['data'].append(pred.detach().cpu().type(torch.float))
            data['targets'].append(labels.detach().cpu().type(torch.long))
            
        data['data'] = torch.cat(data['data'],0)
        data['targets'] = torch.cat(data['targets'],0)
        return data


if __name__ == '__main__':
    vit = VitWrapper('vit_base_patch16_224')
    #data = torch.randn(1, 3, 224, 224)
    #out = vit.model(data)
    #print(out.shape)