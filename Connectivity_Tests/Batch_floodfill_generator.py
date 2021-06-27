import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import concurrent.futures


def floodfill_data_pair(_):
    # Load in training image
    training_img = cv2.imread('ti.png', 0)

    # Randomly pick a part of the image
    window_size = 128
    top_left_row_coord = np.random.randint(training_img.shape[0] - window_size + 1)
    top_left_col_coord = np.random.randint(training_img.shape[1] - window_size + 1)

    # Extract window
    img_channels = training_img[top_left_row_coord:top_left_row_coord + window_size,
             top_left_col_coord:top_left_col_coord + window_size]

    # Prep test image
    img_channels[img_channels == 255] = 1
    # print(img_channels.shape)
    img_channels = cv2.pyrDown(img_channels)
    img_channels = cv2.pyrDown(img_channels)
    # print(img_channels.shape)

    # Prep seed image
    img_seed = np.zeros_like(img_channels)
    kernel = np.ones((3, 3), np.uint8)  # flow in all directions
    img_seed = cv2.dilate(img_seed, kernel, iterations=2)

    # Pick a random point in a channel
    while True:
        y = np.random.randint(img_channels.shape[0])
        x = np.random.randint(img_channels.shape[1])
        if img_channels[y, x] == 1:
            img_seed[y, x] = 1
            break

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

    # Make a binary version
    img_sum_clip = img_sum * 1.0
    img_sum_clip[img_sum_clip != 0] = 1

    # Return Results
    return img_seed, img_channels, img_sum_clip


def generate_training_batch(batch_size=64):
    data_x = np.zeros((batch_size, 2, 32, 32))
    data_y = np.zeros((batch_size, 1, 32, 32))
    for i in range(batch_size):
        data_x[i, 0], data_x[i, 1], data_y[i, 0] = floodfill_data_pair()

    return data_x, data_y


def generate_training_batch_mp(batch_size=64):
    data_x = np.zeros((batch_size, 2, 32, 32))
    data_y = np.zeros((batch_size, 1, 32, 32))
    # breakpoint()
    num_list = np.arange(64)
    mp_results_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:  # 14.5
        mp_results_list = executor.map(floodfill_data_pair, num_list)
    mp_results_list = list(mp_results_list)
    mp_results_list = np.asarray(mp_results_list)

    data_x = mp_results_list[:, 0:2, :, :]
    data_y = mp_results_list[:, 2:3, :, :]

    return data_x, data_y


# For testing purposes only
if __name__ == "__main__":

    # Get data pair
    # seed, channels, clip = floodfill_data_pair()
    # x, y = generate_training_batch()
    # print(x.shape)
    # print(y.shape)
    generate_training_batch_mp(batch_size=64)

    # Show results
    # plt.matshow(seed)
    # plt.matshow(channels)
    # plt.matshow(clip)
    # plt.show()
