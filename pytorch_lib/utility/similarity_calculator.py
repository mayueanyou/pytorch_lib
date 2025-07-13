import os,sys,torch
from ..dataset_lib.dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class SimilarityCalculator():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.distence_weight = 1.001
    
    def p_norm(self,label_features,input_features,p):
        distance = torch.cdist(input_features,label_features,p=p)
        return distance

    def convert_distance_to_similarity(self,distance):
        similarity = torch.neg(distance) + (torch.max(distance)*self.distence_weight)
        similarity = similarity / (torch.max(similarity)*self.distence_weight)
        return similarity

    def convert_Cos_similarity_to_distance(self,similarity_raw):
        return 1 - similarity_raw
    
    def mul(self,label_features,input_features):
        similarity = input_features @ label_features.T
        return similarity

    def cos(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = torch.nn.functional.cosine_similarity(input_features,label_features,dim=2)
        return similarity

    def calculate_similarity(self,label_features,input_features,dis_func='L1'):
        label_features = label_features.to(self.device)
        input_features = input_features.to(self.device)
        if dis_func=='L1': return self.p_norm(label_features,input_features,p=1)
        elif dis_func=='L2': return self.p_norm(label_features,input_features,p=2)
        elif dis_func=='Mul': return self.mul(label_features,input_features)
        elif dis_func=='Cos': return self.cos(label_features,input_features)
        elif type(dis_func)==int or type(dis_func)==float: return self.p_norm(label_features,input_features,dis_func,p=dis_func)
        else: raise Exception('SimilarityCalculator: Key error!')
    
    def __call__(self,label_features,input_features,dis_func='L1',topk=1,batch_mode=False,batch_size = [1000,1000]): #[n,d], [x,d]
        if batch_mode:
            batch_size[0] = len(label_features)if batch_size[0] == -1 else batch_size[0]
            batch_size[1] = len(input_features)if batch_size[1] == -1 else batch_size[1]
            
            label_features_dl = DataLoader(CustomDataset(label_features,None,print_info=False), batch_size=batch_size[0])
            input_features_dl = DataLoader(CustomDataset(input_features,None,print_info=False), batch_size=batch_size[1])
            
            similarity_list = []
            
            for input_features_batch in tqdm(input_features_dl):
                input_features_batch = input_features_batch.to(self.device)
                
                similarity_batch_list = []
                for label_features_batch in label_features_dl:
                    label_features_batch = label_features_batch.to(self.device)
                    
                    similarity = self.calculate_similarity(label_features_batch,input_features_batch,dis_func=dis_func)
                    similarity = similarity.detach().cpu()
                    similarity_batch_list.append(similarity)
                    
                similarity_batch = torch.cat((similarity_batch_list),dim=1)
                similarity_list.append(similarity_batch)
            similarity = torch.cat((similarity_list))
        else:
            similarity = self.calculate_similarity(label_features,input_features,dis_func=dis_func)
        
        similarity_raw = similarity
        if dis_func not in ['Cos','Mul']: similarity = self.convert_distance_to_similarity(similarity)
        
        similarity = similarity.softmax(dim=-1)
        values, indices = similarity.topk(topk)
        return values, indices, similarity, similarity_raw
    
    def only_get_indices(self,label_features,input_features,dis_func='L1',topk=1,batch_mode=False,batch_size = [1000,1000]):
        if batch_mode:
            batch_size[0] = len(label_features)if batch_size[0] == -1 else batch_size[0]
            batch_size[1] = len(input_features)if batch_size[1] == -1 else batch_size[1]
            
            label_features_dl = DataLoader(CustomDataset(label_features,None,print_info=False), batch_size=batch_size[0])
            input_features_dl = DataLoader(CustomDataset(input_features,None,print_info=False), batch_size=batch_size[1])
            
            indices = []
            
            for input_features_batch in tqdm(input_features_dl):
                input_features_batch = input_features_batch.to(self.device)
                
                similarity_batch_list = []
                for label_features_batch in label_features_dl:
                    label_features_batch = label_features_batch.to(self.device)
                    
                    similarity = self.calculate_similarity(label_features_batch,input_features_batch,dis_func=dis_func)
                    similarity = similarity.detach().cpu()
                    similarity_batch_list.append(similarity)
                    
                similarity_batch = torch.cat((similarity_batch_list),dim=1)
                
                if dis_func not in ['Cos','Mul']: similarity_batch = self.convert_distance_to_similarity(similarity_batch)
                similarity_batch = similarity_batch.softmax(dim=-1)
                values, indices_batch = similarity_batch.topk(topk)
                indices.append(indices_batch)
                
            indices = torch.cat((indices))
        else:
            similarity = self.calculate_similarity(label_features,input_features,dis_func=dis_func)
            if dis_func not in ['Cos','Mul']: similarity = self.convert_distance_to_similarity(similarity)
            similarity = similarity.softmax(dim=-1)
            values, indices = similarity.topk(topk)
            
        return indices
    
    def print_top_predictions(self,label, candidate_texts, candidate_indices, candidate_similaritys):
        print(f"Label: {label}",end=' | ')
        for i in range(len(candidate_indices)):
            indice = candidate_indices[i]
            print(f"{i+1} {candidate_texts[indice]}: {100 * candidate_similaritys[indice].item():.2f}%", end=', ')
        print()
    
    def evalue_dataset_print(self,label_features,dataset,dis_func='L1',label_text=None):
        if label_text is None: label_text = [i for i in range(len(label_features))]
        
        values, indices, similarity, similarity_raw = self(label_features,dataset.data,dis_func=dis_func)
        for i in range(len(indices)):
            self.print_top_predictions(label_text[dataset.targets[i].item()], label_text, indices[i], similarity[i])
    
    def evalue_dataset(self,label_features,label_id,dataset,dis_func='L1'):
        data = dataset.data.to(self.device)
        targets = dataset.targets.to(self.device)
        values, indices, similarity, similarity_raw = self(label_features,data,dis_func=dis_func,topk=1)
        indices = indices.flatten()
        indices = label_id[indices]
        accuracy = torch.sum(indices == targets).item() / len(targets)
        return accuracy
    
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
