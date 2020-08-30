

# # Copy files to a directory
# from shutil import copyfile
# src = "copy_file_test.py"
# dst = "ti/copy_file_test.py"
# copyfile(src, dst)

# # # Make a folder if it doesn't exist
# import os
# if not os.path.exists('train_data'):
#     os.makedirs('train_data')

# # # Find and replace text in a file
# import fileinput
# with fileinput.FileInput("train2d_with_conditioning_template.py", inplace=True) as file:
#     for line in file:
#         print(line.replace('batchSize_here', '16'), end='')
# with fileinput.FileInput("train2d_with_conditioning_template.py", inplace=True) as file:
#     for line in file:
#         print(line.replace('lr_here', '0.002'), end='')

# # Run a python script
# import subprocess
# subprocess.run(r'python train2d_with_conditioning.py', shell=True)

# # Generate possible combination (Carteian product)
# import itertools
# list_a = [1, 2, 3]
# list_b = [4, 5, 6, 7]
# for a, b in itertools.product(list_a, list_b):
#     print(a, b)
