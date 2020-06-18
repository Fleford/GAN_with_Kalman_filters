# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:35:10 2020

@author: 101053914
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nnmodels import netD
from nnmodels import netG
import matplotlib.pyplot as plt

nrow = 129 
ncol = 129

nr = 100       # the number of realizations

# Dimension of latent vector
zx = 5
zy = 5
prior_weight = 3
LearningRate = 0.1


# Prepare GAN generator
device = torch.device("cuda:0")
netG = netG(1, 1, 64, 5, 1)
netG.load_state_dict(torch.load('netG_epoch_10.pth'))
netG.to(device)
netG.eval()
torch.set_grad_enabled(True)    # Gradients must be set to true

netD = netD(1, 64, 5, 1)
netD.load_state_dict(torch.load('netD_epoch_10.pth'))
netD.to(device)
netD.eval()
torch.set_grad_enabled(True)    # Gradients must be set to true

# Create mask
Sample_Point = 90
Reference_k = np.loadtxt('Ref_ln_K.txt')
mask_k = np.zeros((nrow, ncol))
Sample = np.zeros((Sample_Point, 2))
for i in range(Sample_Point):
    id_X = np.random.randint(1, nrow)
    id_Y = np.random.randint(1, ncol)
    Sample[i, 0] = id_X
    Sample[i, 1] = id_Y
    mask_k[id_X, id_Y] = 1

# breakpoint()

plt.matshow(Reference_k)
for i in range(0,Sample_Point):                                       # the location of sample points
    plt.scatter(Sample[i, 0], Sample[i, 1], marker='o', c='', edgecolors='r', s = 10)
    
plt.savefig('real_img_k.png')
plt.close()

real_img_k = torch.from_numpy(Reference_k.reshape(1,1,nrow,ncol)).float().to(device)
mask_k = torch.from_numpy(mask_k).float().to(device)

z_optimum = nn.Parameter(torch.rand(nr, 1, zx, zy, device=device)*2-1)
optimizer_z = optim.Adam([z_optimum], lr=LearningRate)

# print()
# print("z_optimum")
# print(z_optimum.shape)
# print(z_optimum)
# print()

#A = netG(z_optimum).squeeze().numpy() 
#Average_A = np.array(A.mean(axis=0))                    # the average of K
#Variance_A = np.array(np.var(A,axis = 0))               # the variance of K
#np.savetxt('Average_Initial.txt', Average_A)
#np.savetxt('Variance_Initial.txt', Variance_A)
#plt.matshow(Average_A)
#plt.savefig('Average_Initial.png')
#plt.close()
#    
#plt.matshow(Variance_A)
#plt.savefig('Variance_Initial.png')
#plt.close()

print("Starting backprop to input ...")
for epoch in range(3):
    optimizer_z.zero_grad()    
    generated_k = netG(z_optimum)

    # print()
    # print("generated_k")
    # print(generated_k.shape)
    # print(generated_k)
    # print()

    K = generated_k
    # k_array = K.cpu().squeeze().numpy()       # variable must be detached first
    # k_array = K.detach().numpy().squeeze()
    K = K.cpu()
    k_array = K.detach().numpy().squeeze()
    Average_K = np.array(k_array.mean(axis=0))                    # the average of K
    Variance_K = np.array(np.var(k_array, axis=0))               # the variance of K
    plt.matshow(k_array[0])
    plt.savefig('k_array[0]_' + str(epoch) + '.png')
    plt.close()
    plt.matshow(Average_K)
    plt.savefig('Average_K_'+str(epoch)+'.png')
    plt.close()
    plt.matshow(Variance_K)
    plt.savefig('Variance_K_'+str(epoch)+'.png')
    plt.close()
    
    discrim_out = netD(generated_k)
    # print()
    # print("discrim_out")
    # print(discrim_out.shape)
    # print(discrim_out)
    # print()
    context_loss = torch.sum(torch.mul((real_img_k-generated_k)**2, mask_k))
    # print()
    # print("(real_img_k-generated_k)**2")
    # print(((real_img_k-generated_k)**2).shape)
    # print((real_img_k-generated_k)**2)
    # print()
    # print()
    # print("mask_k")
    # print(mask_k.shape)
    # print(mask_k)
    # print()
    # print()
    # print("torch.mul((real_img_k-generated_k)**2, mask_k)")
    # print((torch.mul((real_img_k-generated_k)**2, mask_k)).shape)
    # print(torch.mul((real_img_k-generated_k)**2, mask_k))
    # print()
    # print()
    # print("torch.sum(torch.mul((real_img_k-generated_k)**2, mask_k))")
    # print((torch.sum(torch.mul((real_img_k-generated_k)**2, mask_k))).shape)
    # print(torch.sum(torch.mul((real_img_k-generated_k)**2, mask_k)))
    # print()
    # print()
    # print("context_loss")
    # print(context_loss.shape)
    # print(context_loss)
    # print()
    prior_loss = -torch.sum(discrim_out)
    # print()
    # print("prior_loss")
    # print(prior_loss.shape)
    # print(prior_loss)
    # print()
    inpaint_loss = context_loss + prior_weight*prior_loss
    
    # inpaint_loss = inpaint_loss.requires_grad_() # Not necessary
    inpaint_loss.backward()
    optimizer_z.step()
    # print('epoch:', epoch, ' loss:', inpaint_loss)
    print('epoch:', epoch, ' loss:', inpaint_loss, 'context_loss', context_loss, 'prior_loss', prior_loss)
    
    # print()
    # print("z_optimum")
    # print(z_optimum)
    # print()

