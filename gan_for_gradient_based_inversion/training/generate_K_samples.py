from utils import get_texture2D_iter
import numpy as np
from matplotlib import pyplot as plt

import imageio


npx = 129
npy = 129
batch_size = 1
nc = 1

texture_dir = 'C:/Users/Fleford/PycharmProjects/gan_for_gradient_based_inv/training/ti/'
data_iter   = get_texture2D_iter(texture_dir, npx=npx, npy=npy, mirror=False, batch_size=1, n_channel=nc)

for filecount in range(100):
    for data in data_iter:
        print(data.shape)
        data = data.squeeze()
        imageio.imwrite("training_images/" + str(filecount) + ".png", data)
        break
