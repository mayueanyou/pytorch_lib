import os,sys,torch,einops,inspect
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lib import*

class Vit_1(nn.Module):
    def __init__(self,image_size=224,patch_size=56,input_channel=3,att_dim=512,depth=6,heads=8,mlp_dim=3072,num_cls=1000):
        super().__init__()
        self.patch_to_embedding = ImageToPatches(image_size,patch_size,att_dim,patch_projection = 'LinearPatches',input_channel=input_channel)
        self.transformer = Transformer(att_dim,mlp_dim,depth,
                                       AttentionWrap(att_dim,SelfAttention(att_dim,heads,linear_projection={'q':True,'k':True,'v':True,'o':True},increase_dim=True)))
        self.mlp_head = fnn_cell(att_dim,mlp_dim,num_cls)
        self.to_cls_token = nn.Identity()
    
    def forward(self, img, mask=None):
        #p = self.patch_size
        #x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(img)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)