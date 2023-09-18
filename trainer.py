import os,sys,copy,torch,random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class CELoss():
    def __init__(self) -> None:
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        
    def calculate_correct(self,pred,label):
        pred = F.softmax(pred,dim=1)
        return (pred.argmax(1) == label).type(torch.float).sum().item()
    
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

class Trainer():
    def __init__(self,net,train_data=None,test_data=None,validate_data=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.validate_dataloader = validate_data

        self.net = net
        
        self.loss = CELoss()

        self.loss_fn = self.loss.loss_fn
    
    def update_extra_info(self):
        pass
    
    def setup_model(self):
        self.net.net_setup()
    
    def train(self):
        def print_loss(batch,loss):
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"[{current:>5d}/{size:>5d}]",end='')
                print(f"loss: {loss:>7f}")

        size = len(self.train_dataloader.dataset)
        self.setup_model()

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred,feature,loss = self.net.train(X,y,self.loss_fn,True)
            print_loss(batch,loss)
    
    def test(self):
        def evalue(model,dataloader):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.net.eval()
            test_loss, correct = 0, 0

            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)

                    pred,feature = model.net(X)
                    test_loss += self.loss_fn(pred, y).item()
                    correct += self.loss.calculate_correct(pred,y)
                test_loss /= num_batches
                correct /= size

                return correct > model.basic_info['best_validate_accuracy'], correct, test_loss
        
        def print_result(name,accuracy,loss):
            print(f"{name}: \n Accuracy: {(100*accuracy):>0.2f}%, Avg loss: {loss:>8f} \n")
        
        def wrap_val_eval(model):
            update,validate_accuracy, validate_loss = evalue(model,self.validate_dataloader)
            _,test_accuracy, test_loss = evalue(model,self.test_dataloader)
            if update: model.update_best_model(validate_accuracy, test_accuracy, test_loss)
            print_result('Validate',validate_accuracy,validate_loss)
            print_result('Test',test_accuracy,test_loss)
            print_result('Best',model.basic_info['best_test_accuracy'],model.basic_info['best_test_loss'])

        print('net:')
        wrap_val_eval(self.net)
      
    def train_test(self, epochs):
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
            sys.stdout.flush()
            