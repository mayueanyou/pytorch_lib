import os,sys,copy,torch,random
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self,net,train_dataloader=None,test_dataloader=None,validate_dataloader=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.validate_dataloader = validate_dataloader
        self.print_info()

        self.net = net
    
    def print_info(self):
        print(f'batch size: {self.train_dataloader.batch_size}')
        print(f'data dimension: {self.train_dataloader.dataset.data.shape[1:]}')
        print(f'data in total:  train[{len(self.train_dataloader.dataset)}] test[{len(self.test_dataloader.dataset)}] validate[{len(self.validate_dataloader.dataset)}]')
        print(f'batchs in total: train[{len(self.train_dataloader)}] test[{len(self.test_dataloader)}] validate[{len(self.validate_dataloader)}]\n')
        print('='*100)
    
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

            pred,loss = self.net.train(X,y)
            print_loss(batch,loss)
        self.net.lr_scheduler.step()
        print('current learning rate: ',self.net.lr_scheduler.get_last_lr())
    
    def test(self):
        print('net:')
        update,validate_performance, validate_loss = self.net.evalue(self.validate_dataloader)
        _,test_performance, test_loss = self.net.evalue(self.test_dataloader)
        if update: self.net.update_best_model(validate_performance,validate_loss,test_performance,test_loss)
        self.net.loss.print_result(validate_performance,validate_loss,test_performance,test_loss,
                                self.net.basic_info['best_validate_performance'],self.net.basic_info['best_validate_loss'],
                                self.net.basic_info['best_test_performance'],self.net.basic_info['best_test_loss'])
    
    def train_test(self, epochs):
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
            sys.stdout.flush()
            