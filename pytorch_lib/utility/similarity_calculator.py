import os,sys,torch
from pytorch_lib.dataset_lib import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class SimilarityCalculator():
    def __init__(self,topk=1,use_cdist=True,batch_mode=False) -> None:
        self.topk = topk
        self.use_cdist = use_cdist
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.distence_weight = 1.001
        self.batch_mode = batch_mode
        
    def __call__(self,label_features,input_features,dis_func='L1'): #[n,d], [x,d]
        label_features = label_features.to(self.device)
        if dis_func=='L1': return self.p_norm_based(label_features,input_features,p=1)
        elif dis_func=='L2': return self.p_norm_based(label_features,input_features,p=2)
        elif dis_func=='Mul': return self.mul(label_features,input_features)
        elif dis_func=='Cos': return self.cos(label_features,input_features)
        elif type(dis_func)==int or type(dis_func)==float: 
            return self.p_norm_based(label_features,input_features,dis_func,p=dis_func)
        else: raise Exception('SimilarityCalculator: Key error!')
    
    def print_top_predictions(self,label, candidate_texts, candidate_indices, candidate_similaritys):
        print(f"Label: {label}",end=' | ')
        for i in range(len(candidate_indices)):
            indice = candidate_indices[i]
            print(f"{i+1} {candidate_texts[indice]}: {100 * candidate_similaritys[indice].item():.2f}%", end=', ')
        print()
    
    def evalue_dataset(self,label_features,dataset,dis_func='L1',label_text=None):
        if label_text is None: label_text = [i for i in range(len(label_features))]
        
        values, indices, similarity, similarity_raw = self(label_features,dataset.data,dis_func=dis_func)
        for i in range(len(indices)):
            self.print_top_predictions(label_text[dataset.targets[i].item()], label_text, indices[i], similarity[i])
    
    def evalue_datasetloader(self,label_features,dataloader,dis_func='L1',label_text=None):
        if label_text is None: label_text = [i for i in range(len(label_features))]
        
        for features, labels in dataloader:
            values, indices, similarity, similarity_raw = self(label_features,features,dis_func=dis_func)
            for i in range(len(indices)):
                self.print_top_predictions(label_text[labels[i].item()], label_text, indices[i], similarity[i])
    
    def select_from_label_one(self,label_features,input_features,dis_func='L1'):
        indice_list = []
        for i in range(len(input_features)):
            data = input_features[i].unsqueeze(0)
            values, indices, similarity = self(label_features,data,dis_func=dis_func)
            indice_list.append(indices)
        indice_list = torch.cat((indice_list))
        return label_features[indice_list]
    
    def select_from_label(self,label_features,input_features,dis_func='L1'):
        values, indices, similarity, similarity_raw = self(label_features,input_features,dis_func=dis_func)
        return label_features[indices]
    
    def topk_similarity(self,similarity):
        similarity_raw = similarity
        #similarity_raw, _ = torch.topk(similarity,self.topk)
        similarity = similarity.softmax(dim=-1)
        values, indices = similarity.topk(self.topk)
        return values, indices, similarity, similarity_raw

    def p_norm_based(self,label_features,input_features,p):
        if self.batch_mode:
            dataset = CustomDataset(input_features,None,print_info=False)
            dataloader = DataLoader(dataset, batch_size=10240)
            similarity_list = []
            for data in tqdm(dataloader):
                data = data.to(self.device)
                similarity_batch = torch.cdist(data,label_features,p=p)
                similarity_batch = similarity_batch.detach().cpu()
                similarity_list.append(similarity_batch)
            similarity = torch.cat((similarity_list))
        else:
            input_features = input_features.to(self.device)
            similarity = torch.cdist(input_features,label_features,p=p)
        
        similarity = torch.neg(similarity) + (torch.max(similarity)*self.distence_weight)
        similarity = similarity / (torch.max(similarity)*self.distence_weight)
        return self.topk_similarity(similarity)
    
    def mul(self,label_features,input_features):
        similarity = input_features @ label_features.T
        return self.topk_similarity(similarity)
    
    def cos(self,label_features,input_features):
        similarity = self.Cosine_similarity(label_features, input_features)
        return self.topk_similarity(similarity)
    
    def Cosine_similarity(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = torch.nn.functional.cosine_similarity(input_features,label_features,dim=2)
        return similarity
