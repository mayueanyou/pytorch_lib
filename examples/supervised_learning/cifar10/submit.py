import os,sys,subprocess,argparse

file_path=os.path.abspath(__file__)
current_path =  os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")
from pytorch_lib import*
from condor_src import*

def submit_wrap(sub_path,py_arguments):
    cc = CondorController(base_path = current_path + '/condor/', 
                         py_file = current_path + '/main.py',
                         condor_version='py3.10',
                         extra_requirements = '&&(Machine != "SURGE-OG-10-5-141-225")&&(Machine != "SURGE-OG-10-5-137-119")&&(Machine != "SURGE-OG-10-5-141-224")') #&&(Machine != "SURGE-OG-10-5-136-229")
    cc.submit(sub_path = sub_path, py_arguments = py_arguments)
    
def exp1():
    #condor_submit(current_path + base_path + 'ResNet_original' , py_file, '-f main -net ResNet_original','py3.10')
    #condor_submit(current_path + base_path + 'Test' , py_file, '-f main -net Test','py3.10')
    #condor_submit(current_path + base_path + 'att_before_res' , py_file, '-f main -net Test5','py3.10')
    #condor_submit(current_path + base_path + 'att2d_before_res' , py_file, '-f main -net Test6','py3.10')
    #condor_submit(current_path + base_path + 'Vit' , py_file, '-f main -net Vit','py3.10')
    #submit_wrap('Vit_1','-f main -net Vit')
    #submit_wrap('Vit_1_fttt','-f main -net Vit')
    #submit_wrap('Vit_1_tftt','-f main -net Vit')
    #submit_wrap('Vit_1_ttft','-f main -net Vit')
    #submit_wrap('Vit_1_tttf','-f main -net Vit')
    #submit_wrap('Vit_1_fftt','-f main -net Vit')
    #submit_wrap('Vit_1_tfft','-f main -net Vit')
    #submit_wrap('Vit_1_ttff','-f main -net Vit')
    #submit_wrap('Vit_1_tftf','-f main -net Vit')
    #submit_wrap('Vit_1_ftft','-f main -net Vit')
    #submit_wrap('Vit_1_fttf','-f main -net Vit')
    #submit_wrap('Vit_1_ffft','-f main -net Vit')
    #submit_wrap('Vit_1_fftf','-f main -net Vit')
    #submit_wrap('Vit_1_ftff','-f main -net Vit')
    #submit_wrap('Vit_1_tfff','-f main -net Vit')
    #submit_wrap('Vit_1_ffff','-f main -net Vit')
    #submit_wrap('Vit_h8_fftt','-f main -net Vit')
    #submit_wrap('Vit_h8_fftt_c1','-f main -net Vit')
    #submit_wrap('Vit_h8_fftt_cpp','-f main -net Vit')
    #submit_wrap('Vit_2_tt','-f main -net Vit_2')
    #submit_wrap('Vit_2_ft','-f main -net Vit_2')
    #submit_wrap('Vit_2_tf','-f main -net Vit_2')
    #submit_wrap('Vit_2_ff','-f main -net Vit_2')
    #submit_wrap('Vit_3_tt','-f main -net Vit_3')
    #submit_wrap('Vit_3_ft','-f main -net Vit_3')
    #submit_wrap('Vit_3_tf','-f main -net Vit_3')
    #submit_wrap('Vit_3_ff','-f main -net Vit_3')
    #submit_wrap('Vit_1_cnn','-f main -net Vit')
    #submit_wrap('Vit_1_cnn_ind','-f main -net Vit')
    #submit_wrap('Att_1','-f main -net Att')
    #submit_wrap('Att_1_tttt,ft','-f main -net Att')
    #submit_wrap('Vit_p32','-f main -net Vit_p32')
    #submit_wrap('MlpMixter','-f main -net MlpMixter')
    #submit_wrap('FNN1','-f main -net FNN1')
    #submit_wrap('pool1','-f main -net pool1')
    submit_wrap('pool1/new','-f main -net pool1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--function', type=str)
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)()