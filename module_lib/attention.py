import os,sys,torch,einops
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from . import*

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

class SelfAttentionWoQK(nn.Module):
    def __init__(self,dim,channle_in,channle_out,linear_projection={'v':True,'o':True}):
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
    def __init__(self, dim,linear_projection={'q':True,'k':True,'v':True,'o':True}):
        super().__init__()
        self.scale = dim ** -0.5

        self.key = nn.Linear(dim, dim) if linear_projection['k'] else None
        self.query = nn.Linear(dim, dim) if linear_projection['q'] else None
        self.value = nn.Linear(dim, dim) if linear_projection['v'] else None
        self.out = nn.Linear(dim, dim) if linear_projection['o'] else None

    def forward(self, x, mask=None):
        keys = self.key(x) if self.key is not None else x
        queries = self.query(x) if self.query is not None else x
        values = self.value(x) if self.value is not None else x
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, values)
        out = self.out(out) if self.out is not None else out
        return out

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, dim, heads=1,increase_dim=False):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        if increase_dim:
            self.to_qkv = nn.Linear(dim, dim * 3 * heads, bias=False)
            self.to_out = nn.Linear(dim * heads, dim)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
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
        out =  self.to_out(out)
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