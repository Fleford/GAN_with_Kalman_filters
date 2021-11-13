import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet, ConvBlock
import concurrent.futures


def rotate_img(image, angle_degrees):
    # Reflect pad the img
    padded_image = np.pad(image, ((image.shape[0], image.shape[1]),
                                          (image.shape[0], image.shape[1])), 'reflect')

    # Rotate the image
    (h, w) = padded_image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_degrees, 1.0)
    rotated_padded_image = cv2.warpAffine(padded_image, M, (w, h))

    # Clip region of interest
    window_size_1 = image.shape[1]
    window_size_0 = image.shape[0]
    axis_1_center = rotated_padded_image.shape[1] // 2
    axis_0_center = rotated_padded_image.shape[0] // 2
    clipped_rotated_padded_image = rotated_padded_image[axis_0_center - window_size_0 // 2:axis_0_center + window_size_0 // 2,
                                   axis_1_center - window_size_1 // 2:axis_1_center + window_size_1 // 2]

    # Return final image
    return clipped_rotated_padded_image


# Load in training image
training_img = cv2.imread('ti.png', 0)

# Randomly pick a part of the training image
window_size = 512
top_left_row_coord = np.random.randint(training_img.shape[0] - window_size + 1)
top_left_col_coord = np.random.randint(training_img.shape[1] - window_size + 1)

# Extract window
img_channels = training_img[top_left_row_coord:top_left_row_coord + window_size,
               top_left_col_coord:top_left_col_coord + window_size]

# Rotate window
rotated_clipped = rotate_img(img_channels, 45)

plt.matshow(img_channels)
plt.matshow(rotated_clipped)
plt.show()






