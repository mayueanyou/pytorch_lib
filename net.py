import os,sys,torch,random
from torchsummary import summary
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class Net():
    def __init__(self,net,load,model_path,optimizer='Adam') -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():print(torch.cuda.get_device_name(0)) 
        else:print('No GPU')
        self.net = net.to(self.device)

        self.best_validate_accuracy = 0
        self.best_test_accuracy = 0
        self.best_test_loss = 0

        self.base_path = model_path
        self.train_model = True
        self.save_model = True

        self.learning_rate = 0.001
        self.optimizer_select = optimizer
        self.extra_info = {}
        self.best_extra_info = None

        self.load(load)
        if load:
            self.train_model = False
            self.save_model = False

        self.update_optimizer()

    def print_info(self):
        print(self.net.name)
        print(f'best validate accuracy: {(self.best_validate_accuracy*100):>0.2f}%')
        print(f'best test accuracy: {(self.best_test_accuracy*100):>0.2f}%')
        print(f'best test loss: {self.best_test_loss:>8f}')
        print(self.best_extra_info)
        #summary(self.net.cuda(), input_size = self.net.input_size, batch_size = -1)
        print()
    
    def train(self,input_data,label,loss_fn,bp):
        pred,feature = self.net(input_data)
        loss = loss_fn(pred,label)
        if self.train_model and bp:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return pred,feature,loss
   
    def net_setup(self):
        self.net.train() if self.train_model else self.net.eval()
    
    def update_optimizer(self):
        if self.optimizer_select == 'SGD': self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_select == 'Adam': self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=0)
    
    def update_name(self,postfix,load):
        self.net.name = self.net.name + '_' + postfix
        self.load(load)
    
    def update_best_model(self,valid_accuracy,test_accuracy,test_loss):
        self.best_validate_accuracy = valid_accuracy
        self.best_test_accuracy = test_accuracy
        self.best_test_loss = test_loss
        if self.save_model: self.save()
        
    def save(self):
        model_path = self.base_path+'%s.pt'%self.net.name
        data = {'test_accuracy':self.best_test_accuracy,'test_loss':self.best_test_loss,'net':self.net.state_dict(),'optimizer':self.optimizer,
                'validate_accuracy':self.best_validate_accuracy,'learning rate':self.learning_rate,'extra_info':self.extra_info}
        torch.save(data,model_path)
        print('----------------')
        print('save model')
        print('----------------')
    
    def load(self,load_model):
        model_path = self.base_path+'%s.pt'%self.net.name
        if not os.path.isfile(model_path): return
        data = torch.load(model_path ,map_location=self.device)
        self.best_test_accuracy = data['test_accuracy']
        self.best_validate_accuracy  = data['validate_accuracy']
        self.best_test_loss = data['test_loss']
        if 'extra_info' in data.keys():
            self.best_extra_info = data['extra_info']
        if load_model: self.net.load_state_dict(data['net'])
        self.update_optimizer()
        self.print_info()