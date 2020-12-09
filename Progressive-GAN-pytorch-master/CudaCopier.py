from shutil import copyfile
import fileinput

# Program that makes seperate copies of the test program for a specfic cuda node

cuda_num = 1
cuda_suffix = '_cuda' + str(int(cuda_num))

print('train' + cuda_suffix + '.py')
train_src = 'train.py'
train_dst = 'train' + cuda_suffix + '.py'
copyfile(train_src, train_dst)

print('progan_modules' + cuda_suffix + '.py')
progan_modules_src = 'progan_modules.py'
progan_modules_dst = 'progan_modules' + cuda_suffix + '.py'
copyfile(progan_modules_src, progan_modules_dst)

# Replace strings in train_dst
with fileinput.FileInput(train_dst, inplace=True) as file:
    for line in file:
        print(line.replace('from progan_modules import Generator, Discriminator',
                           'from progan_modules' + cuda_suffix + ' import Generator, Discriminator'), end='')
with fileinput.FileInput(train_dst, inplace=True) as file:
    for line in file:
        print(line.replace('return output_matrix.cuda(), torch.as_tensor(random_matrix, dtype=torch.float32, device=device)',
                           'return output_matrix.cuda(device=device), torch.as_tensor(random_matrix, dtype=torch.float32, device=device)'), end='')
with fileinput.FileInput(train_dst, inplace=True) as file:
    for line in file:
        print(line.replace('copy(\'train.py\', log_folder + \'/train_%s.py\' % post_fix)',
                           'copy(\'train' + cuda_suffix + '.py\', log_folder + \'/train_%s.py\' % post_fix)'), end='')
with fileinput.FileInput(train_dst, inplace=True) as file:
    for line in file:
        print(line.replace('copy(\'progan_modules.py\', log_folder + \'/model_%s.py\' % post_fix)',
                           'copy(\'progan_modules' + cuda_suffix + '.py\', log_folder + \'/model_%s.py\' % post_fix)'), end='')
with fileinput.FileInput(train_dst, inplace=True) as file:
    for line in file:
        print(line.replace('one = torch.FloatTensor([1]).to(device)',
                           'one = torch.tensor(1, dtype=torch.float).to(device)'), end='')
with fileinput.FileInput(train_dst, inplace=True) as file:
    for line in file:
        print(line.replace('\'--gpu_id\', type=int, default=0',
                           '\'--gpu_id\', type=int, default=' + str(int(cuda_num))), end='')
