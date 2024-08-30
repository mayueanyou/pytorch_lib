import os,sys,torch
from tqdm import tqdm


class VisionLanguage:
    def __init__(self,text_encoder=None,image_encoder=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
    
    def multiplication_similarity(self,text_features,image_features):
        return 100.0 * image_features @ text_features.T
    
    def L1_similarity(self,text_features,image_features):
        image_features = image_features.unsqueeze(1)
        image_features = image_features.repeat(1,text_features.size(0),1)
        similarity = torch.abs(image_features - text_features)
        similarity = torch.sum(similarity,-1)
        similarity = torch.neg(similarity)
        return similarity
    
    def L2_similarity(self,text_features,image_features):
        image_features = image_features.unsqueeze(1)
        image_features = image_features.repeat(1,text_features.size(0),1)
        similarity = torch.pow(image_features - text_features,2)
        similarity = torch.sum(similarity,-1)
        similarity = torch.neg(similarity)
        return similarity
    
    def get_predictions(self,text_features,image_features,topk=1,dis_func='L1'):
        if dis_func == 'L1': similarity = self.L1_similarity(text_features,image_features)
        if dis_func == 'L2': similarity = self.L2_similarity(text_features,image_features)
        if dis_func == 'Mul': similarity = self.multiplication_similarity(text_features,image_features)
        similarity = similarity.softmax(dim=-1)
        values, indices = similarity.topk(topk)
        #print(indices)
        #input()
        return values, indices
    
    def eval_img_dataset(self,text_features,dataset):
        batch = 0
        acc = 0
        for image_features, labels in tqdm(dataset):
            values, indices = self.get_predictions(text_features,image_features)
            indices = torch.flatten(indices)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)/len(labels)
            batch += 1
        acc /= batch
        print(acc)
    
    def eval_img_dataset_few_shot(self,text_features,dataset):
        batch = 0
        acc = 0
        text_features.targets = torch.flatten(text_features.targets,0)
        text_features = torch.flatten(text_features.data,0,1)
        
        for image_features, labels in tqdm(dataset):
            values, indices = self.get_predictions(text_features,image_features)
            indices = torch.flatten(indices)
            indices = torch.div(indices, 5)
            indices = indices.to(torch.long)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)/len(labels)
            batch += 1
        acc /= batch
        print(acc)