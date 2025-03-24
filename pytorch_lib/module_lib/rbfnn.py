import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFNN(nn.Module):
    def __init__(self, dim_in, dim_out,centers=None):
        super().__init__()
        #self.beta = nn.Parameter(torch.ones(dim_out))
        self.beta = nn.Parameter(torch.ones(dim_out))
        if centers is None: centers = torch.randn(dim_out, dim_in)
        self.centers = nn.Parameter(centers)
        self.distence_weight = 1.001

    def forward(self, x):
        #x_expanded = x.unsqueeze(1)  # (batch_size, 1, in_features)
        #centers_expanded = self.centers.unsqueeze(0)  # (1, out_features, in_features)
        #dist = torch.norm(x_expanded - centers_expanded, dim=2)  # (batch_size, out_features)
        #print(x.shape)
        #print(self.centers.shape)
        #input()
        similarity = torch.cdist(x,self.centers,p=1)
        similarity = torch.neg(similarity) + (torch.max(similarity)*self.distence_weight)
        #similarity = similarity / (torch.max(similarity)*self.distence_weight)
        similarity = self.beta * similarity
        #dist = dist.softmax(dim=1)
        #print(dist.shape)
        #print(dist)
        return similarity