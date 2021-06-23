import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from progan_modules_copy import Generator
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
input_code_size = 128

generator = Generator(in_channel=64, input_code_dim=128, pixel_norm=False, tanh=False).to(device)


# generator.load_state_dict(torch.load('trial_test18_2020-10-12_22_29/checkpoint/160000_g.model'))
generator.load_state_dict(torch.load('trial_test18_2020-10-18_17_37/checkpoint/140000_g.model'))

# sample input data: vector for Generator
gen_z = torch.randn(b_size, input_code_size).to(device)

# generate condition array
data_iter = get_texture2D_iter('ti/', batch_size=b_size)
real_image_raw_res = torch.Tensor(next(data_iter)).to(device)
cond_array, cond_mask = generate_condition(real_image_raw_res)

cond_downsampler = torch.nn.MaxPool2d((8, 8), stride=(8, 8))

# broadcast first cond_array to whole batch
one_cond_array = torch.zeros_like(cond_array)
for slice in range(len(cond_array)):
    one_cond_array[slice] = cond_array[0]
cond_array = one_cond_array

# broadcast first cond array to the whole batch (cond_mask)
one_cond_mask = torch.zeros_like(cond_mask)
for slice in range(len(cond_mask)):
    one_cond_mask[slice] = cond_mask[0]
cond_mask = one_cond_mask

ds_mask = cond_downsampler(cond_mask)

# generate conditioned images
fake_image = generator(gen_z, one_cond_array, step=5, alpha=1.0)

# # plot sample of fake images
plt.matshow(real_image_raw_res[0, 0].cpu().detach().numpy())
plt.matshow(cond_array[0, 0].cpu().detach().numpy())
# plt.matshow(ds_mask[0, 0].cpu().detach().numpy())
plt.matshow(fake_image[0, 0].cpu().detach().numpy())
# plt.matshow(cond_mask[0, 0].cpu().detach().numpy())
# plt.show()

# Calculate statistics
images = fake_image.cpu().squeeze().detach().numpy()
avg_img = np.mean(images, axis=0)
std_img = np.std(images, axis=0)

plt.matshow(avg_img)
plt.matshow(std_img)
plt.show()

breakpoint()