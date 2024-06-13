import os,sys,torch,random,argparse
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=2, threshold=10000, edgeitems=None, linewidth=10000, profile=None, sci_mode=False)

from module import*
import pytorch_template as pt


dataset_path = "~/datasets"
current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")

def main(name):
    dataset = pt.CIFAR10(dataset_path)
    training_data,test_data,validate_data = dataset.loaders()
    net = pt.Net(net = getattr(sys.modules[__name__], name)(),load = False,model_folder_path=current_path+'/model/',loss=CELoss())
    trainer = pt.Trainer(net,training_data,test_data,validate_data)
    trainer.train_test(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="None")
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args.net)