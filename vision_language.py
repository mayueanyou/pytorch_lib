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
        acc,num_data = 0,0
        for image_features, labels in tqdm(dataset):
            values, indices, similarity = self.similarity_calculator(text_features,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            num_data += len(labels)
        
        acc = acc.to(torch.float) / num_data
        print('accuracy: ',acc)
        return acc,num_data
    
    def eval_img_dataset_with_label(self,text_features,test_label,dataset,dis_func='L1'):
        acc,num_data = 0,0
        test_label = test_label.to(self.device)
        for image_features, labels in tqdm(dataset):
            values, indices, similarity = self.similarity_calculator(text_features,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            indices = test_label[indices]
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            num_data += len(labels)
        acc = acc.to(torch.float) / num_data
        print('accuracy: ',acc)
        return acc,num_data
    
    def eval_img_dataset_few_shot(self,text_features,dataset,dis_func='L1'):
        acc = 0
        num_data = 0
        text_features.targets = torch.flatten(text_features.targets,0)
        text_features = torch.flatten(text_features.data,0,1)
        
        for image_features, labels in tqdm(dataset):
            values, indices, similarity= self.similarity_calculator(text_features,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            #indices = torch.div(indices, 5)
            indices = indices.to(torch.long)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            num_data += len(labels)
        
        acc = acc.to(torch.float) / num_data
        print(acc)