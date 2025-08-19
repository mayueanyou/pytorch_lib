import os,sys,torch
from torch import nn

from .criterion import Criterion

class MSELoss(Criterion):
    def __init__(self) -> None:
        self.loss_fn = nn.MSELoss()
    
    def calculate_performance(self,pred,label):
        distance = torch.cdist(pred,label,p=2)
        performance = torch.exp(torch.neg(torch.mean(distance)))
        return performance
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(pred,label)

class MSELoss_Binary(Criterion):
    def __init__(self) -> None:
        self.loss_fn = nn.MSELoss()
        self.threshold = 0.5
    
    def calculate_performance(self,pred,label):
        temp_p = torch.zeros((len(label),1))
        if torch.any(pred>self.threshold):
            temp_p[pred>self.threshold] = 1.0
        pred = temp_p
        correct = ((pred>self.threshold) == (label>self.threshold)).type(torch.float).sum().item()
        return correct
    
    def calculate_loss(self,pred,label):
        loss = self.loss_fn(pred.view(-1),label.float())
        return loss