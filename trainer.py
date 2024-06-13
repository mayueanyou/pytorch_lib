import os,sys,copy,torch,random
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self,net,train_data=None,test_data=None,validate_data=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.validate_dataloader = validate_data

        self.net = net
    
    def update_extra_info(self):
        pass
    
    def train(self):
        def print_loss(batch,loss):
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"[{current:>5d}/{size:>5d}]",end='')
                print(f"loss: {loss:>7f}")

        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred,feature,loss = self.net.train(X,y,True)
            print_loss(batch,loss)
    
    def test(self):
        def print_result(name,accuracy,loss=0):
            print(f"{name}: \n Accuracy: {(100*accuracy):>0.2f}%, Avg loss: {loss:>8f} \n")
        
        def wrap_val_eval(model):
            update,validate_accuracy, validate_loss = self.net.evalue(self.validate_dataloader)
            _,test_accuracy, test_loss = self.net.evalue(self.test_dataloader)
            if update: model.update_best_model(validate_accuracy, test_accuracy, test_loss)
            print_result('Validate',validate_accuracy,validate_loss)
            print_result('Test',test_accuracy,test_loss)
            print_result('Best Validate',model.basic_info['best_validate_accuracy'])
            print_result('Best Test',model.basic_info['best_test_accuracy'],model.basic_info['best_test_loss'])

        print('net:')
        wrap_val_eval(self.net)
      
    def train_test(self, epochs):
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
            sys.stdout.flush()
            