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
prior_weight = 2
LearningRate = 0.1


# Prepare GAN generator
device = torch.device("cpu")
netG = netG(1, 1, 64, 5, 1)
netG.load_state_dict(torch.load('netG_epoch_10.pth'))
netG.to(device)
netG.eval()
torch.set_grad_enabled(False)

netD = netD(1, 64, 5, 1)
netD.load_state_dict(torch.load('netD_epoch_10.pth'))
netD.to(device)
netD.eval()
torch.set_grad_enabled(False)


Sample_Point = 10
Reference_k = np.loadtxt('Ref_ln_K.txt')
mask_k = np.zeros((nrow, ncol))
Sample = np.zeros((Sample_Point, 2))
for i in range(Sample_Point):
    id_X = np.random.randint(1,nrow)
    id_Y = np.random.randint(1,ncol)
    Sample[i,0] =  id_X
    Sample[i,1] =  id_Y
    mask_k[id_X,id_Y] = 1

plt.matshow(Reference_k)
for i in range(0,Sample_Point):                                       # the location of sample points
    plt.scatter(Sample[i,0], Sample[i,1], marker='o',c='', edgecolors='r', s = 10)
    
plt.savefig('real_img_k.png')
plt.close()


real_img_k = torch.from_numpy(Reference_k.reshape(1,1,nrow,ncol)).float()
mask_k = torch.from_numpy(mask_k).float()




z_optimum = nn.Parameter(torch.rand(nr, 1, zx, zy, device=device)*2-1)
optimizer_z = optim.Adam([z_optimum], lr = LearningRate)


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
for epoch in range(10):
    optimizer_z.zero_grad()    
    generated_k = netG(z_optimum)
    
    K = generated_k
    k_array = K.cpu().squeeze().numpy()
    Average_K = np.array(k_array.mean(axis=0))                    # the average of K
    Variance_K = np.array(np.var(k_array,axis = 0))               # the variance of K
    plt.matshow(Average_K)
    plt.savefig('Average_K_'+str(epoch)+'.png')
    plt.close()
    plt.matshow(Variance_K)
    plt.savefig('Variance_K_'+str(epoch)+'.png')
    plt.close()
    
    discrim_out = netD(generated_k)
    context_loss = torch.sum(torch.mul((real_img_k-generated_k)**2,mask_k))
    prior_loss = -torch.sum(discrim_out)
    inpaint_loss = context_loss + prior_weight*prior_loss
    inpaint_loss = inpaint_loss.requires_grad_()
    inpaint_loss.backward()
    optimizer_z.step()
    print('epoch:',epoch,' loss:',inpaint_loss)

