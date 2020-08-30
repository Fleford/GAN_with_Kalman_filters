from shutil import copyfile
import fileinput
import itertools
import subprocess

batchsize_list = ['32', '16']
lr_list = ['0.1', '0.01', '0.001', '0.0001']
cntxt_weight_list = ['0.1', '1.0', '0.01', '1']
beta1_here_list = ['0.25', '0.5', '1.1']
l2_fac_list = ['1e-6', '1e-7', '1e-8']

for batchsize, lr, cntxt_weight, beta1, l2_fac in itertools.product(batchsize_list, lr_list,
                                                                 cntxt_weight_list, beta1_here_list, l2_fac_list):
    # Copy and rename template files
    src = 'train2d_with_conditioning_template.py'
    dst = "train2d_with_conditioning.py"
    copyfile(src, dst)

    # Replace placeholder strings in the files
    with fileinput.FileInput("train2d_with_conditioning.py", inplace=True) as file:
        for line in file:
            print(line.replace('batchSize_here', batchsize), end='')
    with fileinput.FileInput("train2d_with_conditioning.py", inplace=True) as file:
        for line in file:
            print(line.replace('lr_here', lr), end='')
    with fileinput.FileInput("train2d_with_conditioning.py", inplace=True) as file:
        for line in file:
            print(line.replace('cntxt_weight_here', cntxt_weight), end='')
    with fileinput.FileInput("train2d_with_conditioning.py", inplace=True) as file:
        for line in file:
            print(line.replace('beta1_here', beta1), end='')
    with fileinput.FileInput("train2d_with_conditioning.py", inplace=True) as file:
        for line in file:
            print(line.replace('l2_fac_here', l2_fac), end='')

    # Run revised python files
    subprocess.run(r'python train2d_with_conditioning.py', shell=True)
