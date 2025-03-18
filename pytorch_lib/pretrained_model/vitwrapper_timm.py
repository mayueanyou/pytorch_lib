import os,sys,timm,torch
import pytorch_lib as ptl
#from python_lib import*
from tqdm import tqdm
from timm.models import checkpoint_seq
from pprint import pprint

class VitWrapper:
    def __init__(self,weight='vit_base_patch16_224') -> None:
        self.weight = weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = timm.create_model(weight,pretrained=True,num_classes=0)
        self.net.to(self.device)
        ptl.freeze_model(self.net)
        
        self.load_transforms()
        self.display_info()
        self.layer_info = ptl.get_layer_info(self.net)
    
    def display_info(self):
        print("Vit: -", self.weight)
        total_params = ptl.count_parameters(self.net)
        print(f'total parameters: {total_params:,}')
        #print('transfors: -',self.transforms)
        print('='*100)
    
    def load_transforms(self):
        data_config = timm.data.resolve_model_data_config(self.net)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
    
    def forward_features(self,x):
        x = self.net.patch_embed(x)
        x = self.net._pos_embed(x)
        x = self.net.patch_drop(x)
        x = self.net.norm_pre(x)
        if self.net.grad_checkpointing and not torch.jit.is_scripting(): x = checkpoint_seq(self.net.blocks, x)
        else: x = self.net.blocks(x)
        x = self.net.norm(x)
        return x

    def forward_head(self,x):
        x = self.net.pool(x)
        x = self.net.fc_norm(x)
        x = self.net.head_drop(x)
        x = self.net.head(x)
        return x
    
    def inference(self,x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def inference_dataset(self,dataset):
        self.net.eval()
        data  = {'data':[],'targets':[]}
        for images, labels in tqdm(dataset):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.net(images)
            
            data['data'].append(pred.detach().cpu().type(torch.float))
            data['targets'].append(labels.detach().cpu().type(torch.long))
            
        data['data'] = torch.cat(data['data'],0)
        data['targets'] = torch.cat(data['targets'],0)
        return data


if __name__ == '__main__':
    vit = VitWrapper('vit_base_patch16_224')
    data = torch.randn(1, 3, 224, 224)
    #out = vit.inference(data)
    #for it in out:
    #    print(it.shape)
    #print(len(out))
    #print(out.shape)