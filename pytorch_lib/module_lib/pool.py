import os,sys,torch
from torch import nn
from . import*
from ..utility import SimilarityCalculator

class Pool(nn.Module):
    def __init__(self,dim,pool_size,select_num=1,dis_func='Cos',func='mean'):
        super().__init__()
        self.pool = nn.parameter.Parameter(torch.rand(pool_size,dim))
        self.loss = nn.CrossEntropyLoss(label_smoothing=0)
        self.func = func
        self.dic_func = dis_func
        self.sc = SimilarityCalculator(topk = select_num)
    
    def mean(self,x):
        values, indices, similarity = self.sc(self.pool,x,dis_func=self.dic_func)
        self.loss(similarity,indices.flatten())
        p = self.pool[indices]
        
        x = x.unsqueeze(1)
        x = torch.cat((x,p),dim=1)
        x = torch.mean(x,dim=1)
        return x
    
    def forward(self,x):
        if self.func == 'mean': x =  self.mean(x)
        return x