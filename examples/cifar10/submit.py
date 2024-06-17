import os,sys,subprocess,argparse

file_path=os.path.abspath(__file__)
current_path =  os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")
from pytorch_lib import*
from condor_src import*

def exp1():
    py_file = current_path + '/main.py'
    base_path = '/condor/'
    #condor_submit(current_path + base_path + 'ResNet_original' , py_file, '-f main -net ResNet_original','py3.10')
    #condor_submit(current_path + base_path + 'Test' , py_file, '-f main -net Test','py3.10')
    #condor_submit(current_path + base_path + 'Test2' , py_file, '-f main -net Test','py3.10')
    #condor_submit(current_path + base_path + 'Test3' , py_file, '-f main -net Test','py3.10')
    #condor_submit(current_path + base_path + 'Test5' , py_file, '-f main -net Test2','py3.10')
    #condor_submit(current_path + base_path + 'Test6' , py_file, '-f main -net Test3','py3.10')
    #condor_submit(current_path + base_path + 'Test7' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'Test8' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'Test9' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'Test_bn' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'Test_1024' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'Test_1024_nob' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'Test_dim9' , py_file, '-f main -net Test4','py3.10')
    #condor_submit(current_path + base_path + 'att_before_res' , py_file, '-f main -net Test5','py3.10')
    #condor_submit(current_path + base_path + 'att2d_before_res' , py_file, '-f main -net Test6','py3.10')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)()