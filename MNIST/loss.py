import os,sys,torch
from torch import nn
import torch.nn.functional as F

class CELoss():
    def __init__(self) -> None:
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        
    def calculate_correct(self,pred,label):
        pred = F.softmax(pred,dim=1)
        return (pred.argmax(1) == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(pred,label)      

class MSELoss_Binary():
    def __init__(self) -> None:
        self.loss_fn = nn.MSELoss()
        self.threshold = 0.5
    
    def calculate_correct(self,pred,label):
        temp_p = torch.zeros((len(label),1))
        if torch.any(pred>self.threshold):
            temp_p[pred>self.threshold] = 1.0
        pred = temp_p
        correct = ((pred>self.threshold) == (label>self.threshold)).type(torch.float).sum().item()
        return correct
    
    def calculate_loss(self,pred,label):
        loss = self.loss_fn(pred.view(-1),label.float())
        return loss