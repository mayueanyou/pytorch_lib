import os,sys,torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from . import*

class ImageToPatches(nn.Module):
    def __init__(self,image_size,patch_size,dim_out,input_channel=3,position=True,cls_token=True):
        super().__init__()
        num_patche = (image_size // patch_size) ** 2
        patch_dim = input_channel * patch_size ** 2
        self.position = position
        self.cls_token = cls_token
        extra_pos = 0
        if cls_token: 
            self.cls_embedding = nn.Parameter(torch.randn(1, 1, dim_out))
            extra_pos = 1
        
        if position: self.pos_embedding = nn.Parameter(torch.randn(1, num_patche + extra_pos, dim_out))
        
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim,dim_out))
        
    def forward(self,x):
        x = self.patch_to_embedding(x)
        if self.cls_token: x = torch.cat((self.cls_embedding.expand(x.shape[0], -1, -1), x), dim=1)
        if self.position: x  += self.pos_embedding
        return x
        

class VIT(nn.Module):
    def __init__(self,*,image_size,patch_size,input_channel,att_dim,depth,heads,mlp_dim,num_cls):
        super().__init__()
        self.patch_to_embedding = ImageToPatches(image_size,patch_size,att_dim,input_channel=input_channel)
        self.transformer = Transformer(att_dim,mlp_dim,depth,AttentionWrap(att_dim,SelfAttentionMultiHead(att_dim,heads)))
        self.mlp_head = fnn_cell(att_dim,mlp_dim,num_cls)
        self.to_cls_token = nn.Identity()
    
    def forward(self, img, mask=None):
        #p = self.patch_size
        #x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(img)
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)