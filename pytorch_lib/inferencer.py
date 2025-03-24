import os,sys,copy,torch,random
import numpy as np
from tqdm import tqdm
import pytorch_lib as ptl

class Inferencer:
    def __init__(self,net) -> None:
        self.device = ptl.device
        self.net = net
        self.net.eval()
    
    def get_inference_data(self,dataloader):
        data_list = []
        label_list = []
        with torch.no_grad():
            for X, y in tqdm(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.net(X)
                data_list.append(pred.detach().cpu())
                label_list.append(y.detach().cpu())
            data_list = torch.cat((data_list))
            label_list = torch.cat((label_list))
        return data_list,label_list