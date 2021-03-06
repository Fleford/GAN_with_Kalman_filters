import torch
import torch.nn as nn
import numpy as np

from nnmodels import netG

import matplotlib.pyplot as plt

"""
Here's some code to practice loading and using trained GAN models
"""

# Inference device
device = torch.device("cpu")

# NetG parameters
nc = 1  # 'number of channeles in original image space'
nz = 1  # 'number of non-spatial dimensions in latent space z'
ngf = 64    # 'initial number of filters for dis'
gfs = 5    # 'kernel size for gen'
ngpu = 1    # 'number of GPUs to use'

# Input batch parameters
batch_size = 1
zx = 4
zy = 4

# File directory
outf = './train_data'
epoch = 5

# Load model
netG = netG(nc, nz, ngf, gfs, ngpu)
netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (outf, epoch)))
netG.to(device)
netG.eval()
print(netG)
print()

# Load input matrix
noise = torch.rand(batch_size, nz, zx, zy, device=device)*2-1
# input_matrix.to(device)
print("noise matrix:")
print(noise)
print(noise.shape)
print()

# Turn off gradient calculation
torch.set_grad_enabled(False)

# forward run the model
output = netG(noise)
print("Output matrix:")
print(output)
print(output.shape)
print()

numpy_output = output.numpy()
numpy_output = numpy_output.squeeze()
np.savetxt("k_array_ref_gan.txt", numpy_output)
print("numpy_output:")
print(numpy_output)
print(numpy_output.shape)
plt.matshow(numpy_output)
plt.savefig('k_array_ref_gan.png')
plt.show()





