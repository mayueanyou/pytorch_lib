import os,sys,torch
from torch import nn
import torch.nn.functional as F
from pytorch_lib.utility import SimilarityCalculator

class SelfContrastiveLoss():
    def __init__(self,mode_sel=0) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode_list = ['L1','L2','Mul']
        self.mode = self.mode_list[mode_sel]
        self.similarity_calculator = SimilarityCalculator(topk=2)
        self.loss_fn = nn.CosineEmbeddingLoss()
        
    def calculate_correct(self,pred,label):
        pred = pred.detach()
        if self.mode == 'L1': values, indices = self.similarity_calculator.l1(pred,pred)
        if self.mode == 'L2': values, indices = self.similarity_calculator.l2(pred,pred)
        if self.mode == 'Mul': values, indices = self.similarity_calculator.mul(pred,pred)
        indices = indices[:,1]
        indices = torch.flatten(indices)
        pred_result = label[indices]
        return torch.eq(label.to(self.device),pred_result.to(self.device)).sum().item()
    
    def calculate_loss(self,pred,label):
        label_pair = label[label]
        label_new = torch.where(label==label_pair,1,-1)
        return self.loss_fn(pred,pred,label_new)

class ContrastiveLoss():
    def __init__(self,mode_sel=0) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode_list = ['L1','L2','Mul']
        self.mode = self.mode_list[mode_sel]
        self.similarity_calculator = SimilarityCalculator(topk=1)
        self.loss_fn_1 = nn.CosineEmbeddingLoss()
        self.loss_fn_2 = nn.CosineEmbeddingLoss()
        
    def calculate_correct(self,pred,label):
        pred_1,pred_2 = pred
        pred_1,pred_2 = pred_1.detach(),pred_2.detach()
        if self.mode == 'L1': values, indices = self.similarity_calculator.l1(pred_1,pred_2)
        if self.mode == 'L2': values, indices = self.similarity_calculator.l2(pred_1,pred_2)
        if self.mode == 'Mul': values, indices = self.similarity_calculator.mul(pred_1,pred_2)
        indices = torch.flatten(indices)
        label_new =  torch.arange(0, len(indices), 1).to(torch.long)
        return torch.eq(indices.to(self.device),label_new.to(self.device)).sum().item()
    
    def calculate_loss(self,pred,label):
        pred_1,pred_2 = pred
        label_new = torch.ones(len(pred_1))
        loss_fn_all = self.loss_fn_1(pred_1,pred_2,label_new) + self.loss_fn_2(pred_2,pred_1,label_new)
        return loss_fn_all

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

class CosEmLoss():
    def __init__(self) -> None:
        self.loss_fn = nn.CosineEmbeddingLoss()
        
    def calculate_correct(self,pred_1,pred_2,label):
        pred_1 = F.softmax(pred_1,dim=1)
        pred_2 = F.softmax(pred_2,dim=1)
        #loss(input1, input2, target)
        #return (pred.argmax(1) == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        pred_1,pred_2 = pred
        return self.loss_fn(pred_1,pred_2,label)

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