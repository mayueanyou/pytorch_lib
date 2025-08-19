import os,sys,torch
from torch import nn
import torch.nn.functional as F
from pytorch_lib.utility import SimilarityCalculator

from .criterion import Criterion

class CELoss(Criterion):
    def __init__(self,label_smoothing=0) -> None:
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def calculate_performance(self,pred,label):
        pred = F.softmax(pred,dim=1)
        return (pred.argmax(1) == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(pred,label)