import os,sys,torch,random,inspect
from torchsummary import summary
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
#from .utility import*
import pytorch_lib as ptl

class Net():
    def __init__(self,net:torch.nn.Module, 
                 load:bool, 
                 model_folder_path:str,
                 postfix:str=None,
                 optimizer:str='Adam',
                 loss:str=None,
                 lr=0.001,
                 lr_s={'gamma':0.99}) -> None:
        
        self.device = ptl.device

        self.net = net.to(self.device)
        self.loss = loss
        self.lr_s = lr_s
        
        self.net_str = inspect.getsource(type(self.net))
        self.net_name = type(self.net).__name__
        self.loss_name = type(self.loss).__name__
        if postfix is not None: self.net_name = self.net_name + '(' + postfix + ')'
        #summary(self.net, self.net.input_size)
        print('module name: ',self.net_name)
        total_params = ptl.count_parameters(self.net)
        print(f'loss fn: {self.loss_name}')

        self.basic_info = {'best_test_performance':0,
                           'best_test_loss':0,
                           'best_validate_performance':0,
                           'best_validate_loss':0,
                           'optimizer':optimizer,
                           'best_module': self.net_str, 
                           'best_loss_fn':self.loss_name,
                           'learning rate':0,
                           'parameters':total_params}
        
        self.extra_info = {}

        self.model_folder_path = model_folder_path
        
        self.train_model = True
        self.save_model = True

        self.learning_rate = lr
        self.optimizer_select = optimizer

        self.load(load)
        if load:
            self.train_model = False
            self.save_model = False

        self.update_optimizer()
        self.update_lr_scheduler()
    
    
    def model_path(self):
        return self.model_folder_path + '%s.pt'%self.net_name

    def print_info(self):
        print('------------------------','model history','------------------------')
        print(f'best validate performance: {(self.basic_info["best_validate_performance"]*100):>0.2f}%')
        print(f'best validate loss: {self.basic_info["best_validate_loss"]:>8f}')
        print(f'best test performance: {(self.basic_info["best_test_performance"]*100):>0.2f}%')
        print(f'best test loss: {self.basic_info["best_test_loss"]:>8f}')
        print(f'total parameters: {self.basic_info["parameters"]}')
        print(f'loss fn: {self.basic_info["best_loss_fn"]}')
        print(f'optimizer: {self.basic_info["optimizer"]}')
        print(f'best_module: \n{self.basic_info["best_module"]}\n')
        print(f'extra_info: \n{self.extra_info}\n')
    
    def get_parameters(self):
        data_list = []
        for parameter in self.net.parameters():
            data_list.append(parameter)
        return data_list
    
    #------------------------update_function------------------------
    
    def update_loss_object(self,loss):
        self.loss = loss
   
    def update_optimizer(self):
        if self.optimizer_select == 'SGD': self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_select == 'Adam': self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=0)
        #print(self.optimizer)
    
    def update_lr_scheduler(self):
        self.lr_scheduler = lr_scheduler.ExponentialLR(self.optimizer,gamma=self.lr_s['gamma'])
        #self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=50)
    
    def update_name(self,postfix,load):
        self.net_name = self.net_name + '_' + postfix
        self.load(load)
    
    def update_best_model(self,validate_performance,validate_loss,test_performance,test_loss):
        if not self.save_model: return
        self.basic_info["best_validate_performance"] = validate_performance
        self.basic_info["best_validate_loss"] = validate_loss
        self.basic_info["best_test_performance"] = test_performance
        self.basic_info["best_test_loss"] = test_loss
        self.basic_info["best_module"] = self.net_str
        self.basic_info["best_loss_fn"] = self.loss_name
        self.save()
    
     #------------------------save&load------------------------
    
    def save(self):
        data = {'net':self.net.state_dict(),'basic_info':self.basic_info,'extra_info':self.extra_info}
        ptl.save_data(self.model_path(),data)
        print('------------------------','save model','------------------------')
    
    def load(self,load_model):
        print(f'current_module:\n{self.basic_info["best_module"]}\n')
        if not os.path.isfile(self.model_path()): return
        data = torch.load(self.model_path() ,map_location=self.device)
        self.basic_info = data['basic_info']
        self.extra_info = data['extra_info']
        if load_model: 
            self.net.load_state_dict(data['net'])
            print('------------------------','load model','------------------------')
        self.update_optimizer()
        self.print_info()
    
    #------------------------nn_function------------------------
    def train(self,input_data,label):
        self.net.train()
        pred = self.net(input_data)
        loss = self.loss.calculate_loss(pred,label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pred,loss
    
    def evalue(self,dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.net.eval()
        loss, performance = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.net(X)
                loss += self.loss.calculate_loss(pred, y).item()
                performance += self.loss.calculate_performance(pred,y)
            loss /= num_batches
            performance /= size
            return self.loss.is_better(performance,loss,self.basic_info['best_validate_performance'],self.basic_info['best_validate_loss']), performance, loss
            #return correct >= self.basic_info['best_validate_performance'], correct, test_loss
    
    def get_confusion_matrix(self,dataloader:torch.utils.data.DataLoader, classes:{},path:str,name:str=''):
        self.net.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred,feature = self.net(X)
                pred = F.softmax(pred,dim=1)
                pred = pred.argmax(1)
                y_pred.extend(pred)
                y_true.extend(y)
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f'{path}/cf_matrix{name}.png')
    
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
    