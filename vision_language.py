import os,sys,torch
from tqdm import tqdm
from pytorch_lib.utility import SimilarityCalculator


class VisionLanguage:
    def __init__(self,text_encoder=None,image_encoder=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.similarity_calculator = SimilarityCalculator()
        
    def eval_img_dataset(self,text_features,dataset,dis_func='L1'):
        acc,batch = 0,0
        for image_features, labels in tqdm(dataset):
            values, indices = self.similarity_calculator(text_features,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)/len(labels)
            batch += 1
        acc /= batch
        print('accuracy: ',acc)
        return acc
    
    def eval_img_dataset_with_label(self,text_features,test_label,dataset,dis_func='L1'):
        acc,batch = 0,0
        for image_features, labels in tqdm(dataset):
            values, indices = self.similarity_calculator(text_features,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            indices = test_label[indices]
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)/len(labels)
            batch += 1
        acc /= batch
        print('accuracy: ',acc)
        return acc
    
    def eval_img_dataset_few_shot(self,text_features,dataset,dis_func='L1'):
        batch = 0
        acc = 0
        text_features.targets = torch.flatten(text_features.targets,0)
        text_features = torch.flatten(text_features.data,0,1)
        
        for image_features, labels in tqdm(dataset):
            values, indices = self.similarity_calculator(text_features,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            #indices = torch.div(indices, 5)
            indices = indices.to(torch.long)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)/len(labels)
            batch += 1
        acc /= batch
        print(acc)