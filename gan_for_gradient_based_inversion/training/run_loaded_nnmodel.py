import torch
import torch.nn as nn

from nnmodels import netD
from nnmodels import netG

# Inference device
device = torch.device("cuda")

# NetG parameters
nc = 1
nz = 1
ngf = 64
gfs = 10
ngpu = 1

# File directory
outf = './train_data'
epoch = 0

# Load model
netG = netG(nc, nz, ngf, gfs, ngpu)
netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (outf, epoch), map_location="cuda:0"))
netG.to(device)
netG.eval()
# netG.to(device)
print(netG)
print()

# Load input matrix
input_matrix = torch.ones((1, 1, 353, 353))
input_matrix = input_matrix.to(device)
# input_matrix.to(device)
print("Input matrix:")
print(input_matrix)
print(input_matrix.shape)
print()

# forward run the model
output = netG(input_matrix)
print("Output matrix:")
print(output)

# device = torch.device("cuda")
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
# model.to(device)
# # Make sure to call input = input.to(device) on any input tensors that you feed to the model







