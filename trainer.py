import torch,random,os,copy,sys
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torchsummary import summary
import torch.nn.functional as F
from tqdm import tqdm

random.seed(0)
torch.manual_seed(0)

class Net():
    def __init__(self,net,load,model_path,optimizer='SGD') -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.net = net.to(self.device)

        self.best_accuracy = 0
        self.best_loss = 0

        self.base_path = model_path+'/model/'
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
        print(f'best accuracy: {(self.best_accuracy*100):>0.2f}%')
        print(f'best loss: {self.best_loss:>8f}')
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
    
    def update_best_model(self,accuracy,loss):
        self.best_accuracy = accuracy
        self.best_loss = loss
        if self.save_model: self.save()
        
    def save(self):
        model_path = self.base_path+'%s.pt'%self.net.name
        data = {'accuracy':self.best_accuracy,'loss':self.best_loss,'net':self.net.state_dict(),'optimizer':self.optimizer,
                'learning rate':self.learning_rate,'extra_info':self.extra_info}
        torch.save(data,model_path)
    
    def load(self,load_model):
        model_path = self.base_path+'%s.pt'%self.net.name
        if not os.path.isfile(model_path): return
        data = torch.load(model_path ,map_location=self.device)
        self.best_accuracy = data['accuracy']
        self.best_loss = data['loss']
        if 'extra_info' in data.keys():
            self.best_extra_info = data['extra_info']
        if load_model: self.net.load_state_dict(data['net'])
        self.update_optimizer()
        self.print_info()

class Trainer():
    def __init__(self,training_data,test_data,validate_data,net):
        super().__init__()
        self.batch_size = 64
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.train_dataloader = DataLoader(training_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size)
        self.validate_dataloader = DataLoader(validate_data, batch_size=self.batch_size)

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
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss /= num_batches
                correct /= size

                return correct > model.best_accuracy, correct, test_loss
        
        def wrap_val_eval(model):
            update,accuracy, loss = evalue(model,self.validate_dataloader)
            print(f"Validate Error: \n Accuracy: {(100*accuracy):>0.2f}%, Avg loss: {loss:>8f} \n")
            _,accuracy, loss = evalue(model,self.test_dataloader)
            print(f"Test Error: \n Accuracy: {(100*accuracy):>0.2f}%, Avg loss: {loss:>8f} \n")
            if update: model.update_best_model(accuracy,loss)
            print(f"Best Error: \n Accuracy: {(100*model.best_accuracy):>0.2f}%, Avg loss: {model.best_loss:>8f} \n")

        print('net:')
        wrap_val_eval(self.net)
      
    def train_test(self, epochs):
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
            sys.stdout.flush()
            