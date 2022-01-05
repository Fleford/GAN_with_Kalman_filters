import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import torch
import torch.nn.functional as F
from unet import UNet, ConvBlock
import concurrent.futures
from sklearn.metrics import balanced_accuracy_score
import timeit

# Start Timer
start = timeit.default_timer()


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open('statistics_log.txt', 'a') as file:
        print(*args, **kwargs, file=file)


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


def pull_sample_img():
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

    return img_channels


def floodfill_data_pair(angle):

    # # Prepare training image with two imgs superimposed to each other, one 90 degrees of the other
    # img_channels_left_right = pull_sample_img()
    # img_channels_up_down = pull_sample_img().transpose()
    # img_channels = img_channels_left_right + img_channels_up_down
    # img_channels[img_channels == 2] = 1

    # Pull in a sample img
    img_channels = pull_sample_img()

    # Rotate the img
    img_channels = rotate_img(img_channels, angle)

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
    for _ in range(10):
        while True:
            y = np.random.randint(img_channels.shape[0])
            x = np.random.randint(img_channels.shape[1])
            if img_channels[y, x] == 0:
                img_short[y, x] = 1
                break
    img_short = cv2.dilate(img_short, kernel, iterations=5)  # expand blockage size
    img_channels = img_channels + img_short  # load blockage into channel image
    img_channels[img_channels != 0] = 1

    # # Remove all channels (all paths allowed) (COMMENT WHEN UNUSED)
    # img_channels[img_channels == 0] = 1

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


def generate_training_batch(batch_size=32, windowsize=128):
    data_x = np.zeros((batch_size, 2, windowsize, windowsize))
    data_y = np.zeros((batch_size, 1, windowsize, windowsize))
    for i in range(batch_size):
        data_x[i, 0], data_x[i, 1], data_y[i, 0] = floodfill_data_pair()

    return data_x, data_y


def generate_training_batch_mp(batch_size=128, batch_angle=0.0):    # 128
    num_list = np.arange(batch_size)
    angle_list = np.ones_like(num_list) * batch_angle

    with concurrent.futures.ThreadPoolExecutor() as executor:  # 14.5
        mp_results_list = executor.map(floodfill_data_pair, angle_list)
    mp_results_list = list(mp_results_list)
    mp_results_list = np.asarray(mp_results_list)

    data_x = mp_results_list[:, 0:2, :, :]
    data_y = mp_results_list[:, 2:3, :, :]

    return data_x, data_y


if __name__ == "__main__":
    # Clear log file
    with open('statistics_log.txt', 'w') as file:
        print()

    # Print headers in the log file
    print_log('Angle, Mean Balance Accuracy')

    # model list
    model_list = ['model_10909.model']

    for model_name in model_list:
        # Prepare UNet model
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0')
        model = UNet(in_channels=2, n_classes=1, padding=True, up_mode='upconv').to(device)
        print_log('')
        print_log(model_name)
        model.load_state_dict(torch.load(model_name))      # Filepath for the UNet model
        model.zero_grad()

        # Calculate stats for angles 0 to 90
        for angle in np.arange(91):  # 91
            # Prep list of balanced accuracy scores
            bac_list = []

            for i in range(1):     # 25
                with torch.no_grad():
                    # Generate test images
                    x, y = generate_training_batch_mp(batch_angle=angle)
                    x = torch.tensor(x).to(device, dtype=torch.float)  # [N, 1, H, W]
                    y = torch.tensor(y).to(device, dtype=torch.float)  # [N, H, W] with class indices (0, 1)

                    # Run the model
                    prediction = model(x)  # [N, 2, H, W]
                    prediction = F.interpolate(prediction, size=x.shape[2:], mode='nearest')

                    # # Calculate loss
                    # context_loss_array = ((prediction - y) ** 2)
                    # context_loss = torch.sum(context_loss_array).log()
                    # context_loss_print = context_loss.item()
                    # print(context_loss_print)

                    # Calculate balanced accuracy
                    prediction_flat = prediction.flatten().cpu().detach().numpy().round()
                    y_flat = y.flatten().cpu().detach().numpy()
                    # TP = torch.count_nonzero(prediction_flat > thres and y_flat > thres)
                    balanced_accuracy = balanced_accuracy_score(y_flat, prediction_flat)
                    bac_list.append(balanced_accuracy)
                    print(balanced_accuracy)

                    # Show pictures of result
                    seed_img = x[0, 0].cpu().detach().numpy() * 255
                    channels_img = x[0, 1].cpu().detach().numpy() * 255
                    segmented_img = prediction[0, 0].cpu().detach().numpy() * 255
                    annotated_img = y[0, 0].cpu().detach().numpy() * 255
                    train_save_img = np.column_stack((seed_img, channels_img, segmented_img, annotated_img))
                    # plt.imshow(train_save_img)
                    # plt.show()
                    # breakpoint()
                    model_name_clean = model_name.replace('.model', '')
                    cv2.imwrite(model_name_clean + '_' + 'angle_' + str(angle) + '.png', train_save_img)

            # Calculate bac list overall stats
            bac_array = np.array(bac_list)
            print('angle: ', angle)
            print('mean: ', np.mean(bac_array))
            print('std: ', np.std(bac_array))
            print_log(angle, np.mean(bac_array))

        # Stop timer
        stop = timeit.default_timer()

        # Print Time
        print('Time: ', stop - start)

