

# # Copy files to a directory
# from shutil import copyfile
# src = "copy_file_test.py"
# dst = "ti/copy_file_test.py"
# copyfile(src, dst)

# # Make a folder if it doesn't exist
import os
if not os.path.exists('train_data'):
    os.makedirs('train_data')
