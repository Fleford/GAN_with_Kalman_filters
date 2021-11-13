import cv2
import numpy as np
from matplotlib import pyplot as plt

img_filepath = 'left_right_img_samples/left_right_sample_5.png'

src_img = cv2.imread(img_filepath, 0)

# Extract images
img_g_A = src_img[:, 0:src_img.shape[1] // 4]
img_g_B = src_img[:, 1 * src_img.shape[1] // 4: 2 * src_img.shape[1] // 4]
img_g_AB = src_img[:, 2 * src_img.shape[1] // 4: 3 * src_img.shape[1] // 4]
img_g_A_g_B = src_img[:, 3 * src_img.shape[1] // 4: src_img.shape[1]]

# Scale up images
img_g_A = cv2.pyrUp(cv2.pyrUp(img_g_A))
img_g_B = cv2.pyrUp(cv2.pyrUp(img_g_B))
img_g_AB = cv2.pyrUp(cv2.pyrUp(img_g_AB))
img_g_A_g_B = cv2.pyrUp(cv2.pyrUp(img_g_A_g_B))

# Clamp the images
_, img_g_A = cv2.threshold(img_g_A, 127, 255, cv2.THRESH_BINARY)
_, img_g_B = cv2.threshold(img_g_B, 127, 255, cv2.THRESH_BINARY)
_, img_g_AB = cv2.threshold(img_g_AB, 127, 255, cv2.THRESH_BINARY)
_, img_g_A_g_B = cv2.threshold(img_g_A_g_B, 127, 255, cv2.THRESH_BINARY)

# Save images
cv2.imwrite('img_g_A.png', img_g_A)
cv2.imwrite('img_g_B.png', img_g_B)
cv2.imwrite('img_g_AB.png', img_g_AB)
cv2.imwrite('img_g_A_g_B.png', img_g_A_g_B)
