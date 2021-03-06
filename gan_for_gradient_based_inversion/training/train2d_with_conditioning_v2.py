# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""
from __future__ import print_function
import argparse
import os
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchsummary import summary
from nnmodels import netD
from nnmodels import netG_conditioned
import numpy as np
from utils import get_texture2D_iter, zx_to_npx
from shutil import copyfile
import matplotlib.pyplot as plt
import time
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=1, help='number of non-spatial dimensions in latent space z')
parser.add_argument('--zx', type=int, default=5, help='number of grid elements in x-spatial dimension of z')
parser.add_argument('--zy', type=int, default=5, help='number of grid elements in y-patial dimension of z')
parser.add_argument('--zx_sample', type=int, default=4, help='zx for saved image snapshots from G')
parser.add_argument('--zy_sample', type=int, default=4, help='zy for saved image snapshots from G')
parser.add_argument('--nc', type=int, default=1, help='number of channeles in original image space')
parser.add_argument('--ngf', type=int, default=64, help='initial number of filters for dis')
parser.add_argument('--ndf', type=int, default=64, help='initial number of filters for gen')
parser.add_argument('--dfs', type=int, default=5, help='kernel size for dis')
parser.add_argument('--gfs', type=int, default=5, help='kernel size for gen')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--niter', type=int, default=10, help='number of iterations per training epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--l2_fac', type=float, default=1e-7, help='factor for l2 regularization of the weights in G and D')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./train_data', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1978, help='manual seed')
parser.add_argument('--data_iter', default='from_ti', help='way to get the training samples')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
np.random.seed(opt.manualSeed)

print("opt.cuda will be set to true")
opt.cuda = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if opt.cuda else "cpu")

torch.backends.cudnn.enabled = True
cudnn.benchmark = True

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
dfs = int(opt.dfs)
gfs = int(opt.gfs)
nc = int(opt.nc)
# zx = int(opt.zx)
# zy = int(opt.zy)
zx = 5
zy = 5
zx_sample = int(opt.zx_sample)
zy_sample = int(opt.zy_sample)
depth=5
# npx=zx_to_npx(zx, depth)
# npy=zx_to_npx(zy, depth)
npx = 129
npy = 129
batch_size = int(opt.batchSize)

if opt.data_iter=='from_ti':
    # texture_dir='D:/gan_for_gradient_based_inv/training/ti/'
    texture_dir = 'C:/Users/Fleford/PycharmProjects/gan_for_gradient_based_inv/training/ti/'
    data_iter   = get_texture2D_iter(texture_dir, npx=npx, npy=npy, mirror=False, batch_size=batch_size, n_channel=nc)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# netG = netG(nc, nz, ngf, gfs, ngpu)
netG = netG_conditioned()

# netG.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = netD(nc, ndf, dfs, ngpu=1)
# netD.apply(weights_init)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.l2_fac)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.l2_fac)


def generate_condition(input_matrix, density=7):
    # ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(input_matrix, dtype=torch.float32)
    random_matrix = torch.randint_like(ref_k_array, 2)
    for x in range(density):
        random_matrix = random_matrix * torch.randint_like(ref_k_array, 2)
    output_matrix = ref_k_array * random_matrix
    # output_matrix = torch.zeros_like(input_matrix)
    return output_matrix.cuda(), torch.as_tensor(random_matrix, dtype=torch.float32, device=device)


# Generate fixed noise and condition
# input_noise = torch.rand(batch_size, nz, zx, zy, device=device)*2-1
fixed_noise = torch.rand(batch_size, 1, zx, zy, device=device)*2-1
reference_k_array = np.loadtxt("k_array_ref_gan.txt")
fixed_batch_k_array = np.zeros((batch_size, nc, npx, npy))
fixed_batch_k_array[:, :] = reference_k_array
fixed_condition, fixed_condition_mask = generate_condition(fixed_batch_k_array)
# fixed_noise = torch.rand(1, nz, zx_sample, zy_sample, device=device)*2-1
# fixed_noise = input_noise
# fixed_condition = input_condition
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()

# summary(netD, (1, npx, npy))
#
# summary(netG, (nz, zx, zy))

# # Make a folder if it doesn't exist
if not os.path.exists('train_data'):
    os.makedirs('train_data')

# Save training code to results folder
print()
print("Saving train2d_with_conditioning_v2.py")
src = "train2d_with_conditioning_v2.py"
dst = "train_data/train2d_with_conditioning_v2.py"
copyfile(src, dst)

print()
print("Saving nnmodels.py")
src = "nnmodels.py"
dst = "train_data/nnmodels.py"
copyfile(src, dst)

gen_iterations = 0
for epoch in range(opt.nepoch):
    for i, data in enumerate(data_iter):
        if i >= opt.niter:
            break
        f = open(opt.outf+"/training_curve.csv", "a")

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        data_tensor = torch.Tensor(data).to(device)
        # label = torch.full((batch_size*zx*zy,), real_label, device=device)
        output = netD(data_tensor)
        label = torch.full_like(output, real_label, device=device)      # forcing label to match output count
        # print("Ran discriminator")
        # print("data.shape")
        # print(data.shape)
        # print("real_cpu.shape")
        # print(real_cpu.shape)
        # print("output.shape")
        # print(output.shape)
        # print("label.shape")
        # print(label.shape)
        # print()

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()


        # train with fake
        train_density = 10
        noise = torch.rand(batch_size, nz, zx, zy, device=device)*2-1
        noise_condition, condition_mask = generate_condition(data, train_density)
        fake = netG(noise, noise_condition)
        # print("Ran generator")
        # print("noise.shape")
        # print(noise.shape)
        # print("noise_condition.shape")
        # print(noise_condition.shape)
        # print("fake.shape")
        # print(fake.shape)
        # print()

        # label.fill_(fake_label)
        output = netD(fake.detach())
        label = torch.full_like(output, fake_label, device=device)  # forcing label to match output count
        # print("look here again")
        # print(fake.shape)
        # print(output.shape)
        # print(label.shape)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()


         ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)

        # Calculate context loss
        context_loss = torch.sum((fake - data_tensor)**2 * condition_mask)
        context_loss_all = torch.sum(((fake - data_tensor)) ** 2)
        cntxt_loss = context_loss.item()    # For print out of context loss
        cntxt_loss_all = context_loss_all.item()

        # errG = criterion(output, label)
        # errG.backward()
        # D_G_z2 = output.mean().item()
        # optimizerG.step()

        # # Calculate distribution loss (not have the same mean and variance of pixel values
        # fake_flat = fake.reshape(fake.shape[0], fake.shape[1], -1)
        # data_tensor_flat = data_tensor.reshape(data_tensor.shape[0], data_tensor.shape[1], -1)
        # mean_loss = torch.sum((fake_flat.mean(axis=2) - data_tensor_flat.mean(axis=2))**2)
        # std_loss = torch.sum((fake_flat.std(axis=2) - data_tensor_flat.std(axis=2)) ** 2)
        # mean_loss_print = mean_loss.item()
        # std_loss_print = std_loss.item()
        #
        # errG = 2.0 *criterion(output, label) + 1.0 * mean_loss + 1.0 * std_loss
        # errG.backward()
        # D_G_z2 = output.mean().item()
        # optimizerG.step()

        # wt_cntxt = 0
        # if epoch > 30:
        #     wt_cntxt = 1.0

        errG = 2.0 * criterion(output, label) + 1.0 * torch.log(context_loss_all) + 1.0 * context_loss
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        gen_iterations += 1

        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f errG: %.4f '
        #       'errD: %.4f mean_loss: %.4f std_loss: %.4f'
        #          % (epoch, opt.nepoch, i, len(data),
        #          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, errG.data, errD.data,
        #             mean_loss_print, std_loss_print))

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f cntxt_loss: %.4f cntxt_loss_all: %.4f'
                 % (epoch, opt.niter, i, len(data),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, cntxt_loss, cntxt_loss_all))
        if (i+1) % opt.niter == 0:
            fake = netG(fixed_noise, fixed_condition)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

        f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f cntxt_loss: %.4f cntxt_loss_all: %.4f'
                 % (epoch, opt.niter, i, len(data),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, cntxt_loss, cntxt_loss_all))
        f.write('\n')
        f.close()
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

# When finished, rename train_data folder with unique id (epoch number)
src = "train_data"
dst = "train_data" + "_" + str(int(time.time()))
os.rename(src, dst)
