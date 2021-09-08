import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from progan_modules import Generator
from utils import get_texture2D_iter
from matplotlib import pyplot as plt
import cv2

device = 'cuda:0'
# device = 'cpu'
b_size = 32
input_z_channels = 6
# step = 7
alpha = 1.0
A_B_dif_sum_total = 0
A_AB_dif_sum_total = 0
B_AB_dif_sum_total = 0
A_B_dif_array_total = 0
A_AB_dif_array_total = 0
B_AB_dif_array_total = 0

generator = Generator(in_channel=128, input_z_channels=input_z_channels, pixel_norm=False, tanh=False).to(device)

# generator.load_state_dict(torch.load('trial_test18_2021-08-02_21_57/checkpoint/180000_g.model'))  # Step 4
# step = 4
# generator.load_state_dict(torch.load('trial_test18_2021-08-06_10_12/checkpoint/360000_g.model'))    # Step 7
# step = 6

# best so far
generator.load_state_dict(torch.load('trial_test18_2021-08-07_16_10/checkpoint/360000_g.model'))
step = 6

# For control
# generator.load_state_dict(torch.load('trial_test18_2021-07-26_0_52/checkpoint/180000_g.model'))

for x in range(100):
    # Disable gradients
    with torch.no_grad():
        # sample input data: vector for Generator
        # second half of the batch is a top-swapped version of the first half
        gen_z_A = torch.randn(b_size, input_z_channels, 2, 2).to(device)
        gen_z_B = torch.randn(b_size, input_z_channels, 2, 2).to(device)
        gen_z_A_left_half = gen_z_A[:, :, :, 0:gen_z_A.shape[3] // 2]
        gen_z_B_right_half = gen_z_B[:, :, :, gen_z_B.shape[3] // 2:gen_z_B.shape[3]]
        gen_z_AB = torch.cat((gen_z_A_left_half, gen_z_B_right_half), dim=3)
        gen_z = torch.cat((gen_z_A, gen_z_B, gen_z_AB), dim=0)
        # breakpoint()

        # Produce images
        fake_image_A = generator(gen_z_A, step=step, alpha=alpha)
        fake_image_B = generator(gen_z_B, step=step, alpha=alpha)
        fake_image_AB = generator(gen_z_AB, step=step, alpha=alpha)

        # Calculate correlations
        A_B_diff_array = (fake_image_A - fake_image_B)**2
        A_B_dif_sum = torch.sum(A_B_diff_array)
        # print('A_B_dif_sum: ', A_B_dif_sum)

        A_AB_diff_array = (fake_image_A - fake_image_AB)**2
        A_AB_dif_sum = torch.sum(A_AB_diff_array)
        # print('A_AB_dif_sum: ', A_AB_dif_sum)

        B_AB_diff_array = (fake_image_B - fake_image_AB)**2
        B_AB_dif_sum = torch.sum(B_AB_diff_array)
        # print('B_AB_dif_sum: ', B_AB_dif_sum)

        print(x, A_B_dif_sum, A_AB_dif_sum, B_AB_dif_sum)

        A_B_dif_sum_total = A_B_dif_sum_total + A_B_dif_sum.cpu().detach().numpy()
        A_AB_dif_sum_total = A_AB_dif_sum_total + A_AB_dif_sum.cpu().detach().numpy()
        B_AB_dif_sum_total = B_AB_dif_sum_total + B_AB_dif_sum.cpu().detach().numpy()

        A_B_dif_array_total = A_B_dif_array_total + A_B_diff_array.cpu().detach().numpy()
        A_AB_dif_array_total = A_AB_dif_array_total + A_AB_diff_array.cpu().detach().numpy()
        B_AB_dif_array_total = B_AB_dif_array_total + B_AB_diff_array.cpu().detach().numpy()

        # # Show results
        img_g_A = fake_image_A[0, 0].cpu().detach().numpy() * 255
        img_g_B = fake_image_B[0, 0].cpu().detach().numpy() * 255
        img_g_AB = fake_image_AB[0, 0].cpu().detach().numpy() * 255
        img_g_A_g_B = np.concatenate((img_g_A[:, 0:img_g_A.shape[1]//2],
                                      img_g_B[:, img_g_B.shape[1]//2:img_g_B.shape[1]]),  axis=1)
        img_all = np.concatenate((img_g_A, img_g_B, img_g_AB, img_g_A_g_B), axis=1)

        # plt.matshow(img_g_A)
        # print(img_g_A.shape)
        # print(gen_z_A[0, 0])
        # plt.matshow(img_g_B)
        # print(gen_z_B[0, 0])
        # plt.matshow(img_g_AB)
        # print(gen_z_AB[0, 0])
        plt.matshow(img_all)

        cv2.imwrite('left_right_sample.png', img_all)


        # plt.figure()

        # # subplot(r,c) provide the no. of rows and columns
        # f, axarr = plt.subplots(1, 3)
        #
        # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        # axarr[0].imshow(img_g_A)
        # axarr[0].axis('off')
        #
        # axarr[1].imshow(img_g_B)
        # axarr[1].axis('off')
        #
        # axarr[2].imshow(img_g_AB)
        # axarr[2].axis('off')

        plt.show()

print(A_B_dif_sum_total, A_AB_dif_sum_total, B_AB_dif_sum_total)
A_B_img = A_B_dif_array_total / np.max(A_B_dif_array_total)
A_AB_img = A_AB_dif_array_total / np.max(A_AB_dif_array_total)
B_AB_img = B_AB_dif_array_total / np.max(B_AB_dif_array_total)
plt.matshow(np.sum(A_B_img, axis=0)[0])
plt.matshow(np.sum(A_AB_img, axis=0)[0])
plt.matshow(np.sum(B_AB_img, axis=0)[0])
plt.show()
