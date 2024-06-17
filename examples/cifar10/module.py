import os,sys,torch,einops,inspect
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lib import*

class ResNet_original(nn.Module):
    def __init__(self,in_channel=3,class_number=10):
        super().__init__()
        x_bool = True
        xt_bool = False

        self.conv1 = cnn_cell(in_channel,64,3,1,1,bias=False)
        self.layer1 = residual_cell(64, 64, 2, 1,{'x':x_bool,'tx':xt_bool})
        self.layer2 = residual_cell(64, 128, 2, 2,{'x':x_bool,'tx':xt_bool})
        self.layer3 = residual_cell(128, 256, 2, 2,{'x':x_bool,'tx':xt_bool})
        self.layer4 = residual_cell(256, 512, 2, 2,{'x':x_bool,'tx':xt_bool})
        self.fc = nn.Linear(512, class_number)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits#, probas

class Test(nn.Module):
    def __init__(self,basic_dim=512,att_channle=17):
        super().__init__()
        self.name = type(self).__name__
        self.patch = ImageToPatches(image_size=32,patch_size=8,dim_out=basic_dim)
        self.transformer1 = Transformer(17,32,basic_dim,1024)
        self.transformer2 = Transformer(32,64,basic_dim,1024)
        self.transformer3 = Transformer(64,128,basic_dim,1024)
        self.transformer4 = Transformer(128,256,basic_dim,1024)
        self.transformer5 = Transformer(256,512,basic_dim,1024)
        self.transformer6 = Transformer(512,256,basic_dim,1024)
        self.to_cls_token = nn.Identity()
        self.mlp_head = fnn_cell(512,1024,10)
    
    def forward(self,x):
        x = self.patch(x)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        x = self.transformer4(x)
        x = self.transformer5(x)
        x = self.transformer6(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

class Test2(nn.Module):
    def __init__(self,basic_dim=512,att_channle=17):
        super().__init__()
        self.name = type(self).__name__
        self.conv1 = cnn_cell(3,64,3,1,1,bias=False)
        self.layer1 = residual_cell(64, 64, 2, 1)
        self.layer2 = residual_cell(64, 128, 2, 2)
        self.layer3 = residual_cell(128, 256, 2, 2)
        self.layer4 = residual_cell(256, 512, 2, 2)
        
        
        self.att1 = Residual(PreNorm(1024, Attention(64,64)))
        self.att2 = Residual(PreNorm(1024, Attention(64,64)))
        self.att3 = Residual(PreNorm(256, Attention(128,128)))
        self.att4 = Residual(PreNorm(64, Attention(256,256)))
        self.att5 = Residual(PreNorm(16, Attention(512,512)))
        self.fc = nn.Linear(512, 10)
    
    def forward(self,x):
        def cell(x,conv,att):
            x = conv(x)
            h,w = x.shape[-2],x.shape[-1]
            x = x.flatten(-2,-1)
            x = att(x)
            x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
            return x
        
        x = cell(x,self.conv1,self.att1)
        
        x = cell(x,self.layer1,self.att2)
        x = cell(x,self.layer2,self.att3)
        x = cell(x,self.layer3,self.att4)
        x = cell(x,self.layer4,self.att5)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

class Test3(nn.Module):
    def __init__(self,basic_dim=512,att_channle=17):
        super().__init__()
        self.name = type(self).__name__
        self.conv1 = cnn_cell(3,64,3,1,1,bias=False)
        self.layer1 = residual_cell(64, 64, 2, 1)
        self.layer2 = residual_cell(64, 128, 2, 2)
        self.layer3 = residual_cell(128, 256, 2, 2)
        self.layer4 = residual_cell(256, 512, 2, 2)
        
        
        self.att1 = nn.Sequential(PreNorm(1024, Attention(64,128)), PreNorm(1024, Attention(128,64)))
        self.att2 = nn.Sequential(PreNorm(1024, Attention(64,128)),PreNorm(1024, Attention(128,64)))
        self.att3 = nn.Sequential(PreNorm(256, Attention(128,256)),PreNorm(256, Attention(256,128)))
        self.att4 = nn.Sequential(PreNorm(64, Attention(256,512)),PreNorm(64, Attention(512,256)))
        self.att5 = nn.Sequential(PreNorm(16, Attention(512,1024)),PreNorm(16, Attention(1024,512)))
        self.fc = nn.Linear(512, 10)
    
    def forward(self,x):
        def cell(x,conv,att):
            x = conv(x)
            h,w = x.shape[-2],x.shape[-1]
            x = x.flatten(-2,-1)
            x = att(x)
            x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
            return x
        
        x = cell(x,self.conv1,self.att1)
        
        x = cell(x,self.layer1,self.att2)
        x = cell(x,self.layer2,self.att3)
        x = cell(x,self.layer3,self.att4)
        x = cell(x,self.layer4,self.att5)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

class Test4(nn.Module):
    def __init__(self,basic_dim=9,att_channle=17):
        super().__init__()
        self.name = type(self).__name__
        self.patch = ImageToPatches(image_size=32,patch_size=1,dim_out=basic_dim,position=False,cls_token=False)
        self.att1 = AttentionWrap(1024,1024,basic_dim)
        self.att2 = AttentionWrap(1024,512,basic_dim)
        self.att3 = AttentionWrap(512,512,basic_dim)
        self.att4 = AttentionWrap(512,256,basic_dim)
        self.att5 = AttentionWrap(256,256,basic_dim)
        self.att6 = AttentionWrap(256,128,basic_dim)
        self.att7 = AttentionWrap(128,128,basic_dim)
        self.att8 = AttentionWrap(128,64,basic_dim)
        self.att9 = AttentionWrap(64,64,basic_dim)
        self.fc = nn.Linear(64*basic_dim,10)
        
        
    
    def forward(self,x):
        x = self.patch(x)
        x = self.att1(x)
        x = self.att2(x)
        x = self.att3(x)
        x = self.att4(x)
        x = self.att5(x)
        x = self.att6(x)
        x = self.att7(x)
        x = self.att8(x)
        x = self.att9(x)
        x = self.fc(x.flatten(-2,-1))
        return x

class Test5(nn.Module):
    def __init__(self,basic_dim=3,att_channle=17):
        super().__init__()
        self.name = type(self).__name__
        self.patch = ImageToPatches(image_size=32,patch_size=1,dim_out=basic_dim,position=False,cls_token=False)
        self.att1 = nn.Sequential(*([AttentionWrap(1024,1024,basic_dim)]*5))
        self.resnet = ResNet_original()
        
        
    
    def forward(self,x):
        x = self.patch(x)
        x = self.att1(x)
        x = rearrange(x, 'b (h w) c-> b c h w', h=32, w=32)
        x = self.resnet(x)
        return x

class Test6(nn.Module):
    def __init__(self,basic_dim=1,att_channle=17):
        super().__init__()
        self.patch = ImageToPatches(image_size=32,patch_size=1,dim_out=basic_dim,position=False,cls_token=False)
        self.att1 = Attention2d(200,32,32)
        self.resnet = ResNet_original(in_channel=200)
    
    def forward(self,x):
        x = self.patch(x)
        x = rearrange(x, 'b (h w) c-> b c h w', h=32, w=32)
        x = self.att1(x)
        x = self.resnet(x)
        
        return x

