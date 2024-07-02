import os,sys,torch,einops
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from . import *

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim,mlp_dim,depth,att_module):
        super().__init__()
        layers = [nn.Sequential(att_module, Residual(PreNorm(dim, fnn_cell(dim, mlp_dim, dim))))] * depth
        self.layers = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        return self.layers(x)

class AttentionWrap(nn.Module):
    def __init__(self,dim,att_module,residual=True,layernorm=True):
        super().__init__()
        if residual and layernorm: self.attention = Residual(PreNorm(dim, att_module))
        if residual and not layernorm: self.attention = Residual(att_module)
        if not residual and layernorm: self.attention = PreNorm(dim, att_module)
    
    def forward(self, x):
        return self.attention(x)

class SelfAttentionFilter(nn.Module):
    def __init__(self,dim,channle_in,channle_out,linear_projection={'v':False,'o':False}):
        super().__init__()
        self.value = nn.Linear(dim, dim) if linear_projection['v'] else None
        self.out = nn.Linear(dim, dim) if linear_projection['o'] else None
        self.attention = nn.Parameter(torch.randn(channle_out,channle_in))
    
    def forward(self,x):
        x = self.value(x) if self.value is not None else x
        attention = self.attention.softmax(dim=-1)
        attention = attention.expand(x.shape[0],-1,-1)
        out = torch.einsum('bij,bjd->bid', attention, x)
        out = self.out(out) if self.out is not None else out
        return out

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=1,increase_dim=False,linear_projection={'q':True,'k':True,'v':True,'o':True}):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        inner_dim = dim if not increase_dim else dim * heads
        
        self.query = nn.Linear(dim, inner_dim, bias=False) if linear_projection['q'] else None
        self.key = nn.Linear(dim, inner_dim, bias=False) if linear_projection['k'] else None
        self.value = nn.Linear(dim, inner_dim, bias=False) if linear_projection['v'] else None
        self.out = nn.Linear(inner_dim, dim, bias=False) if linear_projection['o'] else None

    def forward(self, x, mask = None):
        q = self.query(x) if self.query is not None else x
        k = self.key(x) if self.key is not None else x
        v = self.value(x) if self.value is not None else x
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attention = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out) if self.out is not None else out
        return out

class SelfAttentionExternalProjection(nn.Module):
    def __init__(self, dim, heads=1,projection={'q':None,'k':None,'v':None,'o':None}):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.query = projection['q']
        self.key = projection['k']
        self.value = projection['v']
        self.out = projection['o']

    def forward(self, x, mask = None):
        q = self.query(x) if self.query is not None else x
        k = self.key(x) if self.key is not None else x
        v = self.value(x) if self.value is not None else x
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attention = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out) if self.out is not None else out
        return out

class Attention2d(nn.Module):
    def __init__(self,channle,h,w):
        super().__init__()
        self.attention = nn.Parameter(torch.randn(channle,h,w))
    
    def forward(self,x):
        attention = self.attention.softmax(dim=-2)
        attention = attention.expand(x.shape[0],-1,-1,-1)
        out = attention * x
        return out