import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from progan_modules import Generator
from utils import get_texture2D_iter
from matplotlib import pyplot as plt


def generate_condition(input_matrix, density=10):
    # ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(input_matrix, dtype=torch.float32)
    random_matrix = torch.randint_like(ref_k_array, 2)
    for x in range(density):
        random_matrix = random_matrix * torch.randint_like(ref_k_array, 2)

    # Enlarge condition points
    sf = 1
    avg_downsampler = torch.nn.MaxPool2d((sf, sf), stride=(sf, sf))
    random_matrix = avg_downsampler(random_matrix)
    random_matrix = F.interpolate(random_matrix, scale_factor=sf, mode="nearest")

    output_matrix = ref_k_array * random_matrix
    # output_matrix = torch.zeros_like(input_matrix)
    return torch.as_tensor(output_matrix, dtype=torch.float32, device=device), torch.as_tensor(random_matrix, dtype=torch.float32, device=device)


device = 'cuda:0'
# device = 'cpu'
b_size = 32
input_z_size = 32

generator = Generator(in_channel=128, input_z_channels=input_z_size, pixel_norm=False, tanh=False).to(device)

generator.load_state_dict(torch.load('trial_test18_2021-06-24_12_58/checkpoint/400000_g.model'))

# Prepare z vectors for training
gen_z_first_half = torch.randn(b_size//2, input_z_size, 2, 2).to(device)
gen_z_top_swap = torch.flip(gen_z_first_half[:, :, 0:gen_z_first_half.shape[2]//2, :], dims=[0])
gen_z_bttm = gen_z_first_half[:, :, gen_z_first_half.shape[2]//2:gen_z_first_half.shape[2], :]
gen_z_second_half = torch.cat((gen_z_top_swap, gen_z_bttm), dim=2)
gen_z = torch.cat((gen_z_first_half, gen_z_second_half), dim=0)

print(gen_z_top_swap.shape)
print(gen_z_bttm.shape)
print(gen_z_first_half[0, 0])
print(gen_z_first_half[-1, 0])
print(gen_z_top_swap[0, 0])
print(gen_z_bttm[0, 0])
print(gen_z_second_half[0, 0])
print(gen_z_second_half.shape)
print(gen_z.shape)
print(gen_z[gen_z.shape[0]//2, 0])


#3 Prep imgs for spatial loss function
gen_imgs_first_half = torch.randn(b_size//2, input_z_size, 4, 4).to(device)
gen_imgs_top_swap = torch.flip(gen_imgs_first_half[:, :, 0:gen_imgs_first_half.shape[2]//2, :], dims=[0])
gen_imgs_bttm = gen_imgs_first_half[:, :, gen_imgs_first_half.shape[2]//2:gen_imgs_first_half.shape[2], :]
gen_imgs_second_half = torch.cat((gen_imgs_top_swap, gen_imgs_bttm), dim=2)
gen_imgs = torch.cat((gen_imgs_first_half, gen_imgs_second_half), dim=0)
# gen_img_true_z_swap = gen_imgs_top_swap_bttm[gen_imgs_top_swap_bttm.shape[0]//2:, :, :, :]

# print(gen_img_true_z_swap.shape)
# print(gen_img_true_z_swap[0, 0])
print(gen_imgs_top_swap.shape)
print(gen_imgs_bttm.shape)
print(gen_imgs_first_half[0, 0])
print(gen_imgs_first_half[-1, 0])
print(gen_imgs_top_swap[0, 0])
print(gen_imgs_bttm[0, 0])
print(gen_imgs_second_half[0, 0])
print(gen_imgs[gen_imgs.shape[0]//2, 0])
print(gen_imgs.shape)

breakpoint()

# sample input data: vector for Generator
gen_z_a = torch.randn(b_size, input_z_size, 2, 2).to(device)
gen_z_b = torch.randn(b_size, input_z_size, 2, 2).to(device)

gen_z_a_rot180 = torch.rot90(gen_z_a, k=2, dims=(2, 3))

# gen_z_a_top = gen_z_a[:,:,0,:]
# gen_z_b_bttm = gen_z_b[:,:,1,:]
# gen_z_ab = torch.stack((gen_z_a_top, gen_z_b_bttm), dim=2)

gen_z_ab = 0.9 * gen_z_a + 0.1 * gen_z_b

print(gen_z_a[0, 0])
print(gen_z_a_rot180[0, 0])
# print(gen_z_b[0, 0])
# print(gen_z_ab[0, 0])

# generate conditioned images
fake_image_a = generator(gen_z_a, step=7, alpha=1.0)
fake_image_a_rot180 = generator(gen_z_a_rot180, step=7, alpha=1.0)
# fake_image_b = generator(gen_z_b, step=7, alpha=1.0)
# fake_image_ab = generator(gen_z_ab, step=7, alpha=1.0)

# # plot sample of fake images
plt.matshow(fake_image_a[0, 0].cpu().detach().numpy())
plt.matshow(fake_image_a_rot180[0, 0].cpu().detach().numpy())
# plt.matshow(fake_image_b[0, 0].cpu().detach().numpy())
# plt.matshow(fake_image_ab[0, 0].cpu().detach().numpy())

plt.show()

breakpoint()