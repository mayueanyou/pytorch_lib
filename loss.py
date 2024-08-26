import os,sys,torch
from torch import nn
import torch.nn.functional as F

class ClipLoss():
    def __init__(self,text_features) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_features = text_features
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        self.loss_fn_text = nn.CrossEntropyLoss(label_smoothing=0)
        self.loss_fn_image = nn.CrossEntropyLoss(label_smoothing=0)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        
    def calculate_correct(self,pred,labels):
        image_features = pred
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(1)
        indices = torch.flatten(indices)
        result = torch.eq(labels.to(self.device),indices.to(self.device))
        acc = torch.sum(result)/len(labels)
        return acc
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(torch.tensor([0]),torch.tensor([0]))
        return self.loss_fn(pred,label)

class CELoss():
    def __init__(self) -> None:
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        
    def calculate_correct(self,pred,label):
        pred = F.softmax(pred,dim=1)
        return (pred.argmax(1) == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(pred,label)

class MSELoss():
    def __init__(self) -> None:
        self.loss_fn = nn.MSELoss()
    
    def calculate_correct(self,pred,label):
        difference = torch.abs(pred - label)
        correct = torch.mean(1 - torch.div(difference,label)) * len(difference)
        return correct
    
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