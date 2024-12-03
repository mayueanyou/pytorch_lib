import os,sys,torch,torchvision
from pytorch_lib import*
from python_lib import*
from tqdm import tqdm

class VitWrapper:
    def __init__(self,model_sel=0,weight_sel=0) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']
        self.model = self.available_models[model_sel]
        self.weights_list = [['IMAGENET1K_V1','IMAGENET1K_SWAG_E2E_V1','IMAGENET1K_SWAG_LINEAR_V1'],
                             ['IMAGENET1K_V1'],
                             ['IMAGENET1K_V1','IMAGENET1K_SWAG_E2E_V1','IMAGENET1K_SWAG_LINEAR_V1'],
                             ['IMAGENET1K_V1'],
                             ['IMAGENET1K_V1'],
                             ['IMAGENET1K_SWAG_E2E_V1','IMAGENET1K_SWAG_LINEAR_V1']]
        self.net = getattr(sys.modules['torchvision.models'],f'{self.model}')(self.weights_list[model_sel][weight_sel])
        self.net.to(self.device)
        self.net.eval()
        print("Vit:",self.model, '-', self.weights_list[model_sel][weight_sel])
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f'total parameters: {total_params:,}')
        
    def evalueate_dataset(self,dataloader):
        acc, batch = 0,0
        data_list = []
        target_list = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.net(images)
                pred = torch.softmax(pred,dim=1)
                acc += (pred.argmax(1) == labels).type(torch.float).sum().item()
                batch += 1
                data_list.append(pred)
                target_list.append(labels)
            acc /= batch
        print('accuracy: ',acc)
        data_list = torch.cat((data_list))
        target_list = torch.cat((target_list))
        return data_list,target_list
    
    def inference_dataset(self,dataset):
        data  = {'data':[],'targets':[]}
        for images, labels in tqdm(dataset):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.net(images)
            
            data['data'].append(pred.cpu().type(torch.float))
            data['targets'].append(labels.cpu().type(torch.long))
        data['data'] = torch.cat(data['data'],0)
        data['targets'] = torch.cat(data['targets'],0)
        return data

if __name__ == "__main__":
    vitwrapper = VitWrapper()