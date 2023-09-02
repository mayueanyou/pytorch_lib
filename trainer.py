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

class Trainer():
    def __init__(self,net,training_data=None,test_data=None,validate_data=None):
        super().__init__()
        self.batch_size = 64
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.train_dataloader = DataLoader(training_data, batch_size=self.batch_size) if training_data is not None else None
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size) if test_data is not None else None
        self.validate_dataloader = DataLoader(validate_data, batch_size=self.batch_size) if validate_data is not None else None

        self.net = net

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
    
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
                    pred = F.softmax(pred,dim=1)
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss /= num_batches
                correct /= size

                return correct > model.best_validate_accuracy, correct, test_loss
        
        def wrap_val_eval(model):
            update,validate_accuracy, validate_loss = evalue(model,self.validate_dataloader)
            print(f"Validate: \n Accuracy: {(100*validate_accuracy):>0.2f}%, Avg loss: {validate_loss:>8f} \n")
            _,test_accuracy, test_loss = evalue(model,self.test_dataloader)
            print(f"Test: \n Accuracy: {(100*test_accuracy):>0.2f}%, Avg loss: {test_loss:>8f} \n")
            if update: model.update_best_model(validate_accuracy,test_accuracy, test_loss)
            print(f"Best: \n Accuracy: {(100*model.best_test_accuracy):>0.2f}%, Avg loss: {model.best_test_loss:>8f} \n")

        print('net:')
        wrap_val_eval(self.net)
      
    def train_test(self, epochs):
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
            sys.stdout.flush()
            