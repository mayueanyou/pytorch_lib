import os,sys,torch 

class SimilarityCalculator():
    def __init__(self,topk=1,use_cdist=True) -> None:
        self.topk = topk
        self.use_cdist = use_cdist
    
    def topk_similarity(self,similarity):
        similarity = similarity.softmax(dim=-1)
        values, indices = similarity.topk(self.topk)
        return values, indices
    
    def p_norm(self,label_features,input_features,p):
        similarity = torch.cdist(input_features,label_features,p=p)
        similarity = torch.neg(similarity)
        return self.topk_similarity(similarity)
    
    def l1(self,label_features,input_features):
        if self.use_cdist: similarity = torch.cdist(input_features,label_features,p=1)
        else: similarity = self.L1_similarity(label_features,input_features)
        similarity = torch.neg(similarity)
        return self.topk_similarity(similarity)
        
    def l2(self,label_features,input_features):
        if self.use_cdist: similarity = torch.cdist(input_features,label_features,p=2)
        else: self.L2_similarity(label_features,input_features)
        similarity = torch.neg(similarity)
        return self.topk_similarity(similarity)
    
    def mul(self,label_features,input_features):
        similarity = input_features @ label_features.T
        return self.topk_similarity(similarity)
    
    def L1_similarity(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = torch.abs(input_features - label_features)
        similarity = torch.sum(similarity,-1)
        return similarity
    
    def L2_similarity(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = torch.pow(input_features - label_features,2)
        similarity = torch.sum(similarity,-1)
        return similarity