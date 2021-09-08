import cv2
import numpy as np
from matplotlib import pyplot as plt

list_of_image_paths = ['left_right_sample_0.png',
                       'left_right_sample_1.png',
                       'left_right_sample_2.png',
                       'left_right_sample_3.png',
                       'left_right_sample_4.png']

# Prep subplots
f, axarr = plt.subplots(len(list_of_image_paths), 4)   # (rows, columns)

# Load in images
for row, image_path in enumerate(list_of_image_paths):
    src_img = cv2.imread('left_right_img_samples/' + image_path, 0)

    # Extract images
    img_g_A = src_img[:, 0:src_img.shape[1]//4]
    img_g_B = src_img[:, 1 * src_img.shape[1]//4: 2 * src_img.shape[1]//4]
    img_g_AB = src_img[:, 2 * src_img.shape[1] // 4: 3 * src_img.shape[1] // 4]
    img_g_A_g_B = src_img[:, 3 * src_img.shape[1] // 4: src_img.shape[1]]

    print(img_g_A.shape)

    # Load images to subplot array
    axarr[row, 0].imshow(img_g_A)
    axarr[row, 0].axis('off')

    axarr[row, 1].imshow(img_g_B)
    axarr[row, 1].axis('off')

    axarr[row, 2].imshow(img_g_AB)
    axarr[row, 2].axis('off')

    axarr[row, 3].imshow(img_g_A_g_B)
    axarr[row, 3].axis('off')

# Add column titles
axarr[0, 0].set_title(r'$G(z_{A})$')
axarr[0, 1].set_title(r'$G(z_{B})$')
axarr[0, 2].set_title(r'$G(z_{A} \oplus z_{B})$')
axarr[0, 3].set_title(r'$G(z_{A}) \oplus G(z_{B})$')

f.tight_layout()
plt.subplots_adjust(left=0.023,
                    bottom=0.031,
                    right=0.977,
                    top=0.923,
                    wspace=0.0,
                    hspace=0.205)
plt.savefig("left_right_samples_fig.png")
plt.show()
