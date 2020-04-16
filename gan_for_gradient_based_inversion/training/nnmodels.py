# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class netD(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dfs = 9, ngpu = 1):
        super(netD, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(

            nn.Conv2d(nc, ndf, dfs, 2, dfs//2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf),

            nn.Conv2d(ndf, ndf*2, dfs, 2, dfs//2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf*2),

            nn.Conv2d(ndf*2, ndf*4, dfs, 2, dfs//2, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf*4),

            nn.Conv2d(ndf*4, ndf*8, dfs, 2, dfs//2, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf*8),

            nn.Conv2d(ndf * 8, 1, kernel_size=dfs, stride=2, padding=2, bias=False),
            nn.Sigmoid()
        )
        self.main = main
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
       
        return output.view(-1, 1).squeeze(1)

#class netD(nn.Module):
#    def __init__(self, nc = 1, ndf = 64, dfs = 9, ngpu = 1):
#        super(netD, self).__init__()
#        self.ngpu = ngpu
#
#        main = nn.Sequential(
#
#            nn.Conv2d(nc, ndf, dfs, 2, dfs//2, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.BatchNorm2d(ndf),
#
#            nn.Conv2d(ndf, ndf*2, dfs, 2, dfs//2, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.BatchNorm2d(ndf*2),
#
#            nn.Conv2d(ndf*2, ndf*4, dfs, 2, dfs//2, bias=False),  
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.BatchNorm2d(ndf*4),
#
#            nn.Conv2d(ndf*4, ndf*8, dfs, 2, dfs//2, bias=False), 
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.BatchNorm2d(ndf*8),
#            
#            nn.Conv2d(ndf * 8, 1, kernel_size=dfs, stride=2, padding=2, bias=False),
#            nn.Sigmoid()
#        )
#        self.main = main
#    
#    def forward(self, input):
#        if input.is_cuda and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input,
#                                               range(self.ngpu))
#        else:
#            output = self.main(input)
#       
#        return output.view(-1, 1).squeeze(1)
    
class netG(nn.Module):
    def __init__(self, nc = 1, nz = 1, ngf = 64, gfs = 5, ngpu = 1):
        super(netG, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(

                nn.ConvTranspose2d(     nz, ngf * 8, gfs, 2, 1, bias=False),
                # nn.ConvTranspose2d(nz, ngf * 8, gfs, 2,  gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 8),

                # Remove later
                nn.ConvTranspose2d(ngf * 8, ngf * 8, gfs, 2, 1, bias=False),
                # nn.ConvTranspose2d(nz, ngf * 8, gfs, 2,  gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 8),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 4),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 2),

                nn.ConvTranspose2d(ngf * 2,     ngf, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf),
               
                nn.ConvTranspose2d(    ngf,      nc, gfs, 2, 2, bias=False),
                nn.ReLU(True),
                
                ### Start dilations ###
                nn.ConvTranspose2d(     nc,ngf, gfs, 1, 6, output_padding=0,bias=False,dilation=3), 
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf),
               
                nn.ConvTranspose2d(    ngf,  nc, gfs, 1, 10, output_padding=0, bias=False,dilation=5),
                
                nn.Tanh()
                
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output

#class netG(nn.Module):
#    def __init__(self, nc = 1, nz = 1, ngf = 64, gfs = 5, ngpu = 1):
#        super(netG, self).__init__()
#        self.ngpu = ngpu
#
#        self.main = nn.Sequential(
#
#                nn.ConvTranspose2d(     nz, ngf * 8, gfs, 2, gfs//2, bias=False), 
#                nn.ReLU(True),
#                nn.BatchNorm2d(ngf * 8),
#
#                nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, 2, gfs//2, bias=False),
#                nn.ReLU(True),
#                nn.BatchNorm2d(ngf * 4),
#
#                nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, 2, gfs//2, bias=False),
#                nn.ReLU(True),
#                nn.BatchNorm2d(ngf * 2),
#
#                nn.ConvTranspose2d(ngf * 2,     ngf, gfs, 2, gfs//2, bias=False),
#                nn.ReLU(True),
#                nn.BatchNorm2d(ngf),
#               
#                nn.ConvTranspose2d(    ngf,      nc, gfs, 2, 2, bias=False),
#                nn.ReLU(True),
#                
#                ### Start dilations ###
#                nn.ConvTranspose2d(     nc, 64, gfs, 1, 6, output_padding=0,bias=False,dilation=3), 
#                nn.ReLU(True),
#                nn.BatchNorm2d(64),
#               
#                nn.ConvTranspose2d(    64,  nc, gfs, 1, 10, output_padding=0, bias=False,dilation=5),
#                nn.Tanh()
#                
#            )
#
#    def forward(self, input):
#        if input.is_cuda and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input,
#                                               range(self.ngpu))
#        else:
#            output = self.main(input)
#        return output


# Below are Fleford's networks

# (Z, X, Y) in, (Y, Z) out
class netG_transformer(nn.Module):
    def __init__(self, nc=2, nz=2, ngf=64, gfs=5, ngpu=1):
        super(netG_transformer, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(

            nn.ConvTranspose2d(2, 2**9, 7, 1, 3, bias=False),
            nn.BatchNorm2d(2**9),
            nn.ReLU(True),

            nn.ConvTranspose2d(2 ** 9, 2 ** 8, 7, 1, 3, bias=False),
            nn.BatchNorm2d(2**8),
            nn.ReLU(True),

            nn.ConvTranspose2d(2 ** 8, 2 ** 7, 7, 1, 3, bias=False),
            nn.BatchNorm2d(2**7),
            nn.ReLU(True),

            nn.ConvTranspose2d(2 ** 7, 2 ** 6, 7, 1, 3, bias=False),
            nn.BatchNorm2d(2**6),
            nn.ReLU(True),

            nn.ConvTranspose2d(2 ** 6, 2 ** 5, 7, 1, 3, bias=False),
            nn.BatchNorm2d(2**5),
            nn.ReLU(True),

            nn.ConvTranspose2d(2 ** 5, 1, 7, 1, 3, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),

            ### Start dilations ###
            nn.ConvTranspose2d(1, 2**9, 5, 1, 6, output_padding=0, bias=False, dilation=3),
            nn.BatchNorm2d(2**9),
            nn.ReLU(True),

            nn.ConvTranspose2d(2**9, 1, 5, 1, 10, output_padding=0, bias=False, dilation=5),

            nn.Tanh()
        )

    def forward(self, input, condition):
        condition = torch.ones_like(input) * condition
        input_to_G = torch.cat((input, condition), 1)
        if input_to_G.is_cuda and self.ngpu > 1:
            output_from_G = nn.parallel.data_parallel(self.main, input_to_G,
                                               range(self.ngpu))
        else:
            output_from_G = self.main(input_to_G)

        output = output_from_G - (abs(condition) * output_from_G) + condition

        return output


# For testing purposes only
if __name__ == "__main__":
    device = "cpu"
    # input_noise = torch.rand(batch_size, nz, zx, zy, device=device) * 2 - 1
    k_matrix = torch.rand(3, 1, 8, 8, device=device) * 2 - 1
    condition_matrix = torch.randint_like(k_matrix, 2) * torch.randint_like(k_matrix, 2)\
                       * torch.randint_like(k_matrix, 2)
    inverse_condition_matrix = torch.ones_like(condition_matrix) - condition_matrix
    input_matrix = torch.cat((k_matrix, condition_matrix), 1)
    # print("input_matrix")
    # print(input_matrix)
    # print(input_matrix.shape)
    print("k_matrix")
    # print(k_matrix)
    print(k_matrix.shape)
    # print("condition_matrix")
    # print(condition_matrix)
    # print(condition_matrix.shape)
    # netG_transformer_1 = netG_transformer()
    # output = netG_transformer_1(k_matrix, condition_matrix)
    # print("netG_transformer_1")
    # print(netG_transformer_1)
    netG = netG(1, 1, 64, 5, 1)
    print(netG)
    output = netG(k_matrix)
    print("output")
    # print(output)
    print(output.shape)

