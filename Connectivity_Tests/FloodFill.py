import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

# Load in test image
img_channels = cv2.imread('test_img.png', 0)
img_channels[img_channels == 255] = 1
print(img_channels.shape)
# img_channels = cv2.pyrDown(img_channels)
print(img_channels.shape)

# Prep seed image
img_seed = np.zeros_like(img_channels)

# Pick a random point in a channel
while True:
    y = np.random.randint(img_channels.shape[0])
    x = np.random.randint(img_channels.shape[1])
    if img_channels[y, x] == 1:
        img_seed[y, x] = 1
        break
kernel = np.ones((3, 3), np.uint8)  # flow in all directions
img_seed = cv2.dilate(img_seed, kernel, iterations=2)
img_dilated = img_seed  # load seed into working array

# Prep sum array
img_sum = np.zeros_like(img_channels) * 1.0
img_sum = img_sum + 1.0 * img_dilated

# Prep dilation kernel
# kernel = np.ones((3, 2), np.uint8)  # flow only left to right
kernel = np.ones((3, 3), np.uint8)  # flow in all directions

# Perform floodfill operation
while True:
    # Take baseline
    num_of_zeros = np.count_nonzero(img_sum)

    # # Run a flood fill iteration
    img_dilated = cv2.dilate(img_dilated, kernel, iterations=1)
    img_dilated = img_dilated * img_channels    # Cut flow at channel edges
    img_sum = img_sum + 1.0 * img_dilated

    # Check if there's no change
    if np.count_nonzero(img_sum) == num_of_zeros:
        break

# Normalize img_sum
img_sum = img_sum / np.max(img_sum)

plt.matshow(img_seed)
plt.matshow(img_channels)
plt.matshow(img_sum)
plt.show()



# Dilate the image


# plt.imshow(img)
# plt.show()

# breakpoint()