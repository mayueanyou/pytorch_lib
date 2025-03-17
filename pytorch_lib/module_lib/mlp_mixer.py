import os,sys,torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from . import*

class Mixer(nn.Module):
    def __init__(self, dim,channel):
        super().__init__()
        self.channel_mixer = nn.Linear(channel,channel)
        self.token_mixer = nn.Linear(dim,dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        #print(x.shape)
        x = rearrange(x, 'b c d -> b d c')
        #print(x.shape)
        #input()
        x = self.channel_mixer(x)
        x = rearrange(x, 'b d c -> b c d') + residual
        x = residual
        x = self.layer_norm(x)
        x = self.token_mixer(x) + residual
        return x