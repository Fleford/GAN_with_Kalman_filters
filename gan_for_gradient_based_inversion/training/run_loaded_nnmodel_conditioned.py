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
epoch = 22

# Load model
# netG = netG(nc, nz, ngf, gfs, ngpu)
netG = netG_transformer()
netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (outf, epoch)))
netG.to(device)
netG.eval()
print(netG)
print()


def generate_condition(ref_k_array):
    # ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(ref_k_array, dtype=torch.float32)
    random_matrix = torch.randint_like(ref_k_array, 2)
    for x in range(4):
        random_matrix = random_matrix * torch.randint_like(ref_k_array, 2)
    output_matrix = ref_k_array * random_matrix
    plt.matshow(output_matrix)

    # output_matrix = torch.zeros_like(input_matrix)
    return output_matrix.cuda()


# Load in reference array
reference_k_array = np.loadtxt("k_array_ref_gan.txt")

# Load input matrix
# noise = torch.rand(batch_size, nz, zx, zy, device=device)*2-1
noise = torch.rand(batch_size, 1, npx, npy, device=device)*2-1
condition = generate_condition(reference_k_array)
# input_matrix.to(device)
print("noise matrix:")
print(noise)
print(noise.shape)
print()

# Turn off gradient calculation
torch.set_grad_enabled(False)

# First run of loop (prepares array for stacking)
# forward run the model
noise = torch.rand(batch_size, 1, npx, npy, device=device)*2-1
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
print()
# plt.matshow(numpy_output)
# plt.show()

# Prepare stack
output_stack = numpy_output

# Following Iterate runs
for runs in range(100):
    # forward run the model
    noise = torch.rand(batch_size, 1, npx, npy, device=device)*2-1
    output = netG(noise, condition)
    output = output.cpu()
    numpy_output = output.numpy()
    numpy_output = numpy_output.squeeze()

    # Append to stack
    output_stack = np.dstack((output_stack, numpy_output))
    print("output_stack")
    print(output_stack.shape)


# Calculate mean and variance
print()
print("mean_array")
mean_array = np.mean(output_stack, axis=2)
print(mean_array)
print(mean_array.shape)
plt.matshow(mean_array)
# plt.show()

print()
print("variance_array")
variance_array = np.var(output_stack, axis=2)
print(variance_array)
print(variance_array.shape)
plt.matshow(variance_array)
# plt.show()

# Show reference k array
ref_k_array = np.loadtxt("k_array_ref_gan.txt")
plt.matshow(ref_k_array)
plt.show()
