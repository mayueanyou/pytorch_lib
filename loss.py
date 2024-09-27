import os,sys,torch
from torch import nn
import torch.nn.functional as F
from pytorch_lib.utility import SimilarityCalculator
from abc import ABC,abstractmethod

class Criterion(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @abstractmethod
    def calculate_correct(self):pass
    
    @abstractmethod
    def calculate_loss(self):pass

class SelfContrastiveLoss(Criterion):
    def __init__(self,mode_sel=0) -> None:
        super().__init__()
        self.mode_list = ['L1','L2','Mul']
        self.mode = self.mode_list[mode_sel]
        self.similarity_calculator = SimilarityCalculator(topk=2)
        self.loss_fn = nn.CosineEmbeddingLoss()
        
    def calculate_correct(self,pred,label):
        pred = pred.detach()
        values, indices = self.similarity_calculator(pred,pred,dis_func=self.mode)
        indices = indices[:,1]
        indices = torch.flatten(indices)
        pred_result = label[indices]
        return torch.eq(label.to(self.device),pred_result.to(self.device)).sum().item()
    
    def calculate_loss(self,pred,label):
        label_1 = label.repeat(len(label))
        pred_1 = pred.repeat(len(label),1)
        label_2 = label.repeat(len(label),1).T.flatten()
        pred_2 = pred.repeat(1,len(label))
        pred_2 = pred_2.view(-1,pred.shape[-1])
        label_new = torch.where(label_1==label_2,1,-1)
        return self.loss_fn(pred_1,pred_2,label_new)

class CE_SimilarityLoss(Criterion):
    def __init__(self,target_tensor,dis_func='L2') -> None: 
        super().__init__()
        self.dis_func = dis_func
        self.target_tensor = target_tensor.to(self.device)
        self.similarity_calculator = SimilarityCalculator(topk=1)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
    
    def calculate_correct(self,pred,label):
        values, indices, similarity = self.similarity_calculator(self.target_tensor,pred,dis_func=self.dis_func)
        indices = torch.flatten(indices)
        return (indices == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        values, indices, similarity = self.similarity_calculator(self.target_tensor,pred,dis_func="L2")
        return self.loss_fn(similarity,label)

class CELoss_SelfContrastiveLoss(Criterion):
    def __init__(self,rate=0.9) -> None:
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        self.em_loss = SelfContrastiveLoss()
        self.rate_ce = rate
        self.rate_em = 1-rate
        
    def calculate_correct(self,pred,label):
        pred_1,pred_2 = pred
        pred_1 = F.softmax(pred_1,dim=1)
        return (pred_1.argmax(1) == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        pred_1,pred_2 = pred
        loss = self.ce_loss_fn(pred_1,label) * self.rate_ce + self.em_loss.calculate_loss(pred_2,label) * self.rate_em
        return loss

class ContrastiveLoss(Criterion):
    def __init__(self,mode_sel=0) -> None:
        super().__init__()
        self.mode_list = ['L1','L2','Mul']
        self.mode = self.mode_list[mode_sel]
        self.similarity_calculator = SimilarityCalculator(topk=1)
        self.loss_fn_1 = nn.CosineEmbeddingLoss()
        self.loss_fn_2 = nn.CosineEmbeddingLoss()
        
    def calculate_correct(self,pred,label):
        pred_1,pred_2 = pred
        pred_1,pred_2 = pred_1.detach(),pred_2.detach()
        values, indices = self.similarity_calculator(pred_1,pred_2,dis_func=self.mode)
        indices = torch.flatten(indices)
        label_new =  torch.arange(0, len(indices), 1).to(torch.long)
        return torch.eq(indices.to(self.device),label_new.to(self.device)).sum().item()
    
    def calculate_loss(self,pred,label):
        pred_1,pred_2 = pred
        label_new = torch.ones(len(pred_1))
        loss_fn_all = self.loss_fn_1(pred_1,pred_2,label_new) + self.loss_fn_2(pred_2,pred_1,label_new)
        return loss_fn_all

class ContrastiveClsLoss(Criterion):
    def __init__(self,mode_sel=0) -> None:
        super().__init__()
        self.mode_list = ['L1','L2','Mul']
        self.mode = self.mode_list[mode_sel]
        self.similarity_calculator = SimilarityCalculator(topk=2)
        self.loss_fn_1 = nn.CosineEmbeddingLoss()
        self.loss_fn_2 = nn.CosineEmbeddingLoss()
        
    def calculate_correct(self,pred,label):
        pred_1,pred_2 = pred
        pred_1,pred_2 = pred_1.detach(),pred_2.detach()
        values, indices = self.similarity_calculator(pred_1,pred_2,dis_func=self.mode)
        indices = torch.flatten(indices)
        label_new =  torch.arange(0, len(indices), 1).to(torch.long)
        return torch.eq(indices.to(self.device),label_new.to(self.device)).sum().item()
    
    def calculate_loss(self,pred,label):
        pred_1,pred_2 = pred
        label_new = torch.ones(len(pred_1))
        loss_fn_all = self.loss_fn_1(pred_1,pred_2,label_new) + self.loss_fn_2(pred_2,pred_1,label_new)
        return loss_fn_all

class ClipLoss(Criterion):
    def __init__(self,text_features) -> None:
        super().__init__()
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

class CosEmLoss(Criterion):
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

class CELoss(Criterion):
    def __init__(self) -> None:
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        
    def calculate_correct(self,pred,label):
        pred = F.softmax(pred,dim=1)
        return (pred.argmax(1) == label).type(torch.float).sum().item()
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(pred,label)

class MSELoss(Criterion):
    def __init__(self) -> None:
        self.loss_fn = nn.MSELoss()
    
    def calculate_correct(self,pred,label):
        difference = torch.abs(pred - label)
        correct = torch.mean(1 - torch.div(difference,label)) * len(difference)
        return correct
    
    def calculate_loss(self,pred,label):
        return self.loss_fn(pred,label)     

class MSELoss_Binary(Criterion):
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