import os,sys,torch
from torch import nn
from . import*

class Fnn(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        layers = [nn.Linear(dim_in,dim_hidden),nn.GELU(),nn.Linear(dim_hidden,dim_out)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layers(x)

def fnn_cell(dim_in,dim_hidden,dim_out):
    layers = [nn.Linear(dim_in,dim_hidden),nn.GELU(),nn.Linear(dim_hidden,dim_out)]
    return nn.Sequential(*layers)