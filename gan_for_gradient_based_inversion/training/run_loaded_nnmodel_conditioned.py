import torch
import torch.nn as nn
import numpy as np

from nnmodels import netG
from nnmodels import netG_transformer

import matplotlib.pyplot as plt

"""
Here's some code to practice loading and using trained GAN models
"""

# Inference device
device = torch.device("cuda:0")

# NetG parameters
nc = 1  # 'number of channeles in original image space'
nz = 1  # 'number of non-spatial dimensions in latent space z'
ngf = 64    # 'initial number of filters for dis'
gfs = 5    # 'kernel size for gen'
ngpu = 1    # 'number of GPUs to use'

# Input batch parameters
batch_size = 1
zx = 97
zy = 97
npx = zx
npy = zy
# File directory
outf = './train_data'
epoch = 26

# Load model
# netG = netG(nc, nz, ngf, gfs, ngpu)
netG = netG_transformer()
netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (outf, epoch)))
netG.to(device)
netG.eval()
print(netG)
print()


def generate_condition(input_matrix):
    ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(ref_k_array, dtype=torch.float32)
    random_matrix = torch.randint_like(ref_k_array, 2)
    for x in range(8):
        random_matrix = random_matrix * torch.randint_like(ref_k_array, 2)
    output_matrix = ref_k_array * random_matrix

    # output_matrix = torch.zeros_like(input_matrix)
    return output_matrix.cuda()


# Load input matrix
# noise = torch.rand(batch_size, nz, zx, zy, device=device)*2-1
noise = torch.rand(batch_size, 1, npx, npy, device=device)*2-1
condition = generate_condition(noise)
# input_matrix.to(device)
print("noise matrix:")
print(noise)
print(noise.shape)
print()

# Turn off gradient calculation
torch.set_grad_enabled(False)

# forward run the model
output = netG(noise, condition)
print("Output matrix:")
print(output)
print(output.shape)
print()

output = output.cpu()
numpy_output = output.numpy()
numpy_output = numpy_output.squeeze()
print("numpy_output:")
print(numpy_output)
print(numpy_output.shape)
plt.matshow(numpy_output)
plt.show()





