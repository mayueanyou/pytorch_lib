import os,sys,torch,random
from torchsummary import summary
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class Net():
    def __init__(self,net,load,model_folder_path,optimizer='Adam') -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():print(torch.cuda.get_device_name(0)) 
        else:print('No GPU')

        self.net = net.to(self.device)
        #summary(self.net, self.net.input_size)
        total_params = sum(p.numel() for p in self.net.parameters())
        print('module name: ',self.net.name)
        print(f'total parameters: {total_params:,}')

        self.basic_info = {'best_test_accuracy':0,'best_test_loss':0,'optimizer':optimizer,
                           'best_validate_accuracy':0,'learning rate':0,'parameters':total_params}
        
        self.extra_info = {}

        self.model_folder_path = model_folder_path
        if not os.path.exists(model_folder_path): os.makedirs(model_folder_path)
        
        self.train_model = True
        self.save_model = True

        self.learning_rate = 0.001
        self.optimizer_select = optimizer


        self.load(load)
        if load:
            self.train_model = False
            self.save_model = False

        self.update_optimizer()
    
    def model_path(self):
        return self.model_folder_path + '%s.pt'%self.net.name

    def print_info(self):
        print(f'best validate accuracy: {(self.basic_info["best_validate_accuracy"]*100):>0.2f}%')
        print(f'best test accuracy: {(self.basic_info["best_test_accuracy"]*100):>0.2f}%')
        print(f'best test loss: {self.basic_info["best_test_loss"]:>8f}')
        print(f'total parameters: {self.basic_info["parameters"]:,}')
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
    
    def update_best_model(self,validate_accuracy,test_accuracy,test_loss):
        if not self.save_model: return
        self.basic_info["best_validate_accuracy"] = validate_accuracy
        self.basic_info["best_test_accuracy"] = test_accuracy
        self.basic_info["best_test_loss"] = test_loss
        self.save()
        
    def save(self):
        data = {'net':self.net.state_dict(),'basic_info':self.basic_info,'extra_info':self.extra_info}
        torch.save(data,self.model_path())
        print('----------------','save model','----------------')
    
    def load(self,load_model):
        if not os.path.isfile(self.model_path()): return
        data = torch.load(self.model_path() ,map_location=self.device)
        self.basic_info = data['basic_info']
        self.best_extra_info = data['extra_info']
        if load_model: 
            self.net.load_state_dict(data['net'])
            print('----------------','load model','----------------')
        self.update_optimizer()
        self.print_info()