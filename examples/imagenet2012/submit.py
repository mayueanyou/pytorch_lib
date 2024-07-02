import os,sys,subprocess,argparse

file_path=os.path.abspath(__file__)
current_path =  os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")
from pytorch_lib import*
from condor_src import*

def submit_wrap(folder,argument):
    py_file = current_path + '/main.py'
    base_path = '/condor/'
    condor_submit(current_path + base_path + folder , py_file, argument,'py3.10','&&(Machine != "SURGE-OG-10-5-141-225")')
    
def exp1():
    submit_wrap('Vit_1','-f main -net Vit_1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)()