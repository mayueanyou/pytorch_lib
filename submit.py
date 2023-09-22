import os,sys,subprocess

def create_folder(path):
    if not os.path.exists(path):os.makedirs(path)

def get_sub_str(arguments):
    lines = ['executable = /home/yma183/tools/conda_wrapper\n']
    lines.append('arguments = %s\n'%arguments)
    lines.append('error= ./err\noutput= ./out\nlog= ./log\n')
    lines.append('Requirements = (TotalGPUs > 0)\n+request_gpus = 1\nqueue')
    return lines

def condor_submit(path,arguments):
    create_folder(path)
    with open(path+"/sub", "w") as file:
        file.writelines(get_sub_str(arguments))
    subprocess.run(["rm log & condor_submit sub"], cwd=path, shell=True)

if __name__ == '__main__':
    file_path=os.path.abspath(__file__)
    current_path =  os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")
    py_file = current_path + '/main.py '
    base_path = '/condor/original/RN_original/'
    #condor_submit(current_path + base_path + 'ResNet_wo_pooling_1' , py_file + '-f train_original -c 1 -net ResNet_wo_pooling -p 1')
