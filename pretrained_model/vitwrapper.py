import os,sys,torch,torchvision
from pytorch_lib import*
from python_lib import*
from tqdm import tqdm
import torchvision.models as tvmodel

class VitWrapper:
    def __init__(self,model_sel=0,weight_sel=0) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']
        self.model = self.available_models[model_sel]
        self.weights_list = [['IMAGENET1K_V1','IMAGENET1K_SWAG_E2E_V1','IMAGENET1K_SWAG_LINEAR_V1']]
        self.net = getattr(sys.modules['torchvision.models'],f'{self.model}')(self.weights_list[model_sel][weight_sel])
        

if __name__ == "__main__":
    vitwrapper = VitWrapper()