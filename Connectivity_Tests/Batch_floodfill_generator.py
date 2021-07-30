import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import concurrent.futures


def floodfill_data_pair_two_points(_):
    # Load in training image
    training_img = cv2.imread('ti.png', 0)

    # Randomly pick a part of the training image
    window_size = 512
    top_left_row_coord = np.random.randint(training_img.shape[0] - window_size + 1)
    top_left_col_coord = np.random.randint(training_img.shape[1] - window_size + 1)

    # Extract window
    img_channels = training_img[top_left_row_coord:top_left_row_coord + window_size,
             top_left_col_coord:top_left_col_coord + window_size]

    # Prep test image
    img_channels[img_channels == 255] = 1

    # Scale down the image
    img_channels = cv2.pyrDown(img_channels)
    img_channels = cv2.pyrDown(img_channels)

    # Prep seed image
    img_seed_1 = np.zeros_like(img_channels)

    # Pick a random point in a channel
    while True:
        y = np.random.randint(img_channels.shape[0])
        x = np.random.randint(img_channels.shape[1])
        if img_channels[y, x] == 1:
            img_seed_1[y, x] = 1
            break

    kernel = np.ones((3, 3), np.uint8)  # flow in all directions
    img_seed_1 = cv2.dilate(img_seed_1, kernel, iterations=2)   # expand seed size
    img_dilated = img_seed_1  # load seed into working array

    # Introduce a blockage
    img_blockage = np.ones_like(img_channels)
    for _ in range(1):
        while True:
            y = np.random.randint(img_channels.shape[0])
            x = np.random.randint(img_channels.shape[1])
            if img_channels[y, x] == 1:
                img_blockage[y, x] = 0
                break
    img_blockage = cv2.erode(img_blockage, kernel, iterations=5)  # expand blockage size
    img_channels = img_channels * img_blockage  # load blockage into channel image

    # Introduce channel shorts
    img_short = np.zeros_like(img_channels)
    for _ in range(32):
        while True:
            y = np.random.randint(img_channels.shape[0])
            x = np.random.randint(img_channels.shape[1])
            if img_channels[y, x] == 0:
                img_short[y, x] = 1
                break
    img_short = cv2.dilate(img_short, kernel, iterations=5)  # expand blockage size
    img_channels = img_channels + img_short  # load blockage into channel image
    img_channels[img_channels != 0] = 1

    # Prep sum array
    img_sum = np.zeros_like(img_channels) * 1.0
    img_sum = img_sum + 1.0 * img_dilated

    # Prep dilation kernel
    kernel = np.ones((3, 3), np.uint8)  # flow in all directions
    # kernel = np.ones((3, 2), np.uint8)  # flow in all directions

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
    img_1_sum = img_sum
    # # Normalize img_sum
    # img_1_sum = img_1_sum / np.max(img_1_sum)

    # Make a binary version
    img_1_sum_clip = img_1_sum * 1.0
    img_1_sum_clip[img_1_sum_clip != 0] = 1

    # # Dist map for first point
    # plt.matshow(img_1_sum)
    # plt.show()

    # # Make dist map for second point
    # Prep seed image for second point
    img_seed_2 = np.zeros_like(img_channels)

    # Pick a random point in a channel
    while True:
        y = np.random.randint(img_channels.shape[0])
        x = np.random.randint(img_channels.shape[1])
        if img_channels[y, x] == 1:
            img_seed_2[y, x] = 1
            break

    kernel = np.ones((3, 3), np.uint8)  # flow in all directions
    img_seed_2 = cv2.dilate(img_seed_2, kernel, iterations=2)  # expand seed size
    img_dilated = img_seed_2  # load seed into working array

    # Prep sum array
    img_sum = np.zeros_like(img_channels) * 1.0
    img_sum = img_sum + 1.0 * img_dilated

    # Prep dilation kernel
    kernel = np.ones((3, 3), np.uint8)  # flow in all directions
    # kernel = np.ones((3, 2), np.uint8)  # flow in all directions

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
    img_2_sum = img_sum
    # # Normalize img_sum
    # img_2_sum = img_2_sum / np.max(img_2_sum)

    # # Find shortest path between points
    img_1_2_sum = img_1_sum + img_2_sum

    max_val = np.max([np.max(img_1_2_sum * img_seed_1), np.max(img_1_2_sum * img_seed_2)])
    min_val = np.min([np.max(img_1_2_sum * img_seed_1), np.max(img_1_2_sum * img_seed_2)])
    # print(max_val)
    # print(min_val)
    # print(np.max(img_1_sum * img_seed_2))
    # print(np.max(img_2_sum * img_seed_1))
    img_1_2_sum_clipped = img_1_2_sum.copy()
    img_1_2_sum_clipped[img_1_2_sum_clipped != max_val] = 0
    img_1_2_sum_clipped[img_1_2_sum_clipped != min_val] = 0
    img_1_2_sum_clipped = img_1_2_sum_clipped + (img_seed_1 * img_1_sum) + (img_seed_2 * img_2_sum)
    img_1_2_sum_clipped[img_1_2_sum_clipped != 0] = 1
    img_1_2_sum_clipped = img_1_2_sum_clipped * img_channels

    img_seeds = img_seed_1 + img_seed_2

    # plt.matshow(img_1_sum)
    # plt.matshow(img_2_sum)
    # plt.matshow(img_1_2_sum)
    # plt.matshow(img_1_2_sum * img_seed_1)
    # plt.matshow(img_1_2_sum * img_seed_2)
    # plt.matshow(img_1_2_sum_clipped)
    # plt.matshow(img_channels)
    # plt.matshow(img_seeds)
    # plt.show()

    # breakpoint()

    # Return Results
    return img_seeds, img_channels, img_1_2_sum_clipped


def floodfill_data_pair(_):
    # Load in training image
    training_img = cv2.imread('ti.png', 0)

    # Randomly pick a part of the training image
    window_size = 512
    top_left_row_coord = np.random.randint(training_img.shape[0] - window_size + 1)
    top_left_col_coord = np.random.randint(training_img.shape[1] - window_size + 1)

    # Extract window
    img_channels = training_img[top_left_row_coord:top_left_row_coord + window_size,
             top_left_col_coord:top_left_col_coord + window_size]

    # Prep test image
    img_channels[img_channels == 255] = 1

    # Scale down the image
    img_channels = cv2.pyrDown(img_channels)
    img_channels = cv2.pyrDown(img_channels)

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
    img_seed = cv2.dilate(img_seed, kernel, iterations=2)   # expand seed size
    img_dilated = img_seed  # load seed into working array

    # Introduce a blockage
    img_blockage = np.ones_like(img_channels)
    for _ in range(16):
        while True:
            y = np.random.randint(img_channels.shape[0])
            x = np.random.randint(img_channels.shape[1])
            if img_channels[y, x] == 1:
                img_blockage[y, x] = 0
                break
    img_blockage = cv2.erode(img_blockage, kernel, iterations=5)  # expand blockage size
    img_channels = img_channels * img_blockage  # load blockage into channel image

    # Prep sum array
    img_sum = np.zeros_like(img_channels) * 1.0
    img_sum = img_sum + 1.0 * img_dilated

    # Prep dilation kernel
    kernel = np.ones((3, 3), np.uint8)  # flow in all directions
    # kernel = np.ones((3, 2), np.uint8)  # flow in all directions

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
    # generate_training_batch_mp(batch_size=64)

    img_seed, img_channels, img_sum_clip = floodfill_data_pair_two_points(1)
    # Show results
    print(img_sum_clip.shape)
    plt.matshow(img_seed)
    plt.matshow(img_channels)
    plt.matshow(img_sum_clip)
    plt.show()
