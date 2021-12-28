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


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open('training_log.txt', 'a') as file:
        print(*args, **kwargs, file=file)


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


def floodfill_data_pair(_):

    # Randomly decide if the image should be just a horz img, or a horz and vert img superimposed and randomly rotated
    train_img_choice = np.random.randint(0, 2)
    if train_img_choice == 0:
        # Prepare training image with two imgs superimposed to each other, one 90 degrees of the other
        img_channels_left_right = pull_sample_img()
        img_channels_up_down = pull_sample_img().transpose()
        img_channels = img_channels_left_right + img_channels_up_down
        img_channels[img_channels == 2] = 1

        # Randomly rotate the img
        img_channels = rotate_img(img_channels, np.random.randint(0, 361))

    if train_img_choice == 1:
        # Pull in a sample img
        img_channels = pull_sample_img()

    # # Randomly decide if the image should by rotated by 90 degrees
    # if np.random.randint(0, 2):
    #     img_channels = np.rot90(img_channels)

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


def generate_training_batch_mp(batch_size=32):
    num_list = np.arange(batch_size)

    with concurrent.futures.ThreadPoolExecutor() as executor:  # 14.5
        mp_results_list = executor.map(floodfill_data_pair, num_list)
    mp_results_list = list(mp_results_list)
    mp_results_list = np.asarray(mp_results_list)

    data_x = mp_results_list[:, 0:2, :, :]
    data_y = mp_results_list[:, 2:3, :, :]

    return data_x, data_y


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    model = UNet(in_channels=2, n_classes=1, padding=True, up_mode='upconv').to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.0, 0.99))
    epochs = 1000000

    m = nn.Sigmoid()
    bce = nn.BCELoss(reduction='sum')

    sf_shrink = 0.4
    best_loss = np.inf
    best_train_val_score = 0

    # Clear log file
    with open('training_log.txt', 'w') as file:
        print()

    # Write header line in log file
    print_log('epoch, loss_print, context_loss_print, error_ratio_print, pos_only_ratio_print')

    x_anim, y_anim = generate_training_batch_mp()
    x_anim = torch.tensor(x_anim).to(device, dtype=torch.float)  # [N, 1, H, W]
    y_anim = torch.tensor(y_anim).to(device, dtype=torch.float)  # [N, H, W] with class indices (0, 1)
    x_anim = torch.rot90(x_anim, k=2, dims=[2, 3])
    y_anim = torch.rot90(y_anim, k=2, dims=[2, 3])


    for epoch in range(epochs):
        x, y = generate_training_batch_mp()
        x = torch.tensor(x).to(device, dtype=torch.float)  # [N, 1, H, W]
        # x = F.interpolate(x, scale_factor=sf_shrink, mode='bilinear')
        # x = F.interpolate(x, size=(500, 500), mode='bilinear')
        y = torch.tensor(y).to(device,  dtype=torch.float)  # [N, H, W] with class indices (0, 1)
        # y = F.interpolate(y, scale_factor=sf_shrink, mode='nearest')
        # y = F.interpolate(y, size=(496, 496), mode='nearest')
        y_null = torch.zeros_like(y)
        y_rand = torch.rand_like(y)

        model.zero_grad()
        prediction = model(x)  # [N, 2, H, W]
        prediction = F.interpolate(prediction, size=x.shape[2:], mode='nearest')

        context_loss_array = ((prediction - y) ** 2)
        context_loss = torch.sum(context_loss_array).log()

        error_ratio = torch.sum(torch.abs(torch.clamp(prediction, min=0.0, max=1.0) - y)) / torch.sum(y)
        error_ratio_print = error_ratio.item()

        pos_only_ratio = torch.sum(y * prediction) / torch.sum(y)
        pos_only_ratio_print = pos_only_ratio.item()

        ref_loss_array = ((y_null - y) ** 2)
        ref_loss = torch.sum(ref_loss_array).log()

        # loss = context_loss - ref_loss
        loss = context_loss
        loss_print = loss.item()
        context_loss_print = context_loss.item()
        print(epoch, loss_print, context_loss_print, error_ratio_print)

        # loss_bce = bce(m(prediction), y)
        # loss_bce_print = loss_bce.item()
        # print(epoch, loss_bce_print)

        loss.backward()
        # loss_bce.backward()
        optim.step()

        # if epoch % 1 == 0:
        if loss_print < best_loss:
            best_loss = loss_print
            torch.save(model.state_dict(), f'model_{str(epoch)}.model')

            with torch.no_grad():
                # Save image for training sample
                seed_img = x[0, 0].cpu().detach().numpy() * 255
                channels_img = x[0, 1].cpu().detach().numpy() * 255
                segmented_img = prediction[0, 0].cpu().detach().numpy() * 255
                annotated_img = y[0, 0].cpu().detach().numpy() * 255
                train_save_img = np.column_stack((seed_img, channels_img, segmented_img, annotated_img))
                cv2.imwrite("train_" + str(epoch) + '.png', train_save_img)
                print_log(epoch, loss_print, context_loss_print, error_ratio_print, pos_only_ratio_print)

                # Run a 180-degree validation test
                x_single_rot = torch.rot90(x, k=2, dims=[2, 3])
                y_single_rot = torch.rot90(y, k=2, dims=[2, 3])
                x_single_rot = torch.tensor(x_single_rot).to(device, dtype=torch.float)
                y_single_rot = torch.tensor(y_single_rot).to(device, dtype=torch.float)
                prediction = model(x_single_rot)  # [N, 2, H, W]
                prediction = F.interpolate(prediction, size=x.shape[2:], mode='nearest')

                # Calculate loss
                # val_loss_bce = bce(m(prediction), y_single_rot)
                # val_loss_bce_print = val_loss_bce.item()
                context_loss_array = ((prediction - y) ** 2)
                context_loss = torch.sum(context_loss_array).log()
                ref_loss_array = ((y_null - y) ** 2)
                ref_loss = torch.sum(ref_loss_array).log()
                val_loss = context_loss - ref_loss
                val_loss_print = val_loss.item()

                # Prep to save a validation test image sample
                seed_img = x_single_rot[0, 0].cpu().detach().numpy() * 255
                channels_img = x_single_rot[0, 1].cpu().detach().numpy() * 255
                segmented_img = prediction[0, 0].cpu().detach().numpy() * 255
                annotated_img = y_single_rot[0, 0].cpu().detach().numpy() * 255
                val_save_img = np.column_stack((seed_img, channels_img, segmented_img, annotated_img))
                cv2.imwrite("val_" + str(epoch) + '.png', val_save_img)

                # Run a 90-degree validation test
                x_single_rot90 = torch.rot90(x, k=1, dims=[2, 3])
                y_single_rot90 = torch.rot90(y, k=1, dims=[2, 3])
                x_single_rot90 = torch.tensor(x_single_rot90).to(device, dtype=torch.float)
                y_single_rot90 = torch.tensor(y_single_rot90).to(device, dtype=torch.float)
                prediction = model(x_single_rot90)  # [N, 2, H, W]
                prediction = F.interpolate(prediction, size=x.shape[2:], mode='nearest')

                # Calculate loss
                # val_loss_bce = bce(m(prediction), y_single_rot)
                # val_loss_bce_print = val_loss_bce.item()
                context_loss_array = ((prediction - y) ** 2)
                context_loss = torch.sum(context_loss_array).log()
                ref_loss_array = ((y_null - y) ** 2)
                ref_loss = torch.sum(ref_loss_array).log()
                rot90_val_loss = context_loss - ref_loss
                rot90_val_loss_print = rot90_val_loss.item()

                # Prep to save a validation test image sample
                seed_img = x_single_rot90[0, 0].cpu().detach().numpy() * 255
                channels_img = x_single_rot90[0, 1].cpu().detach().numpy() * 255
                segmented_img = prediction[0, 0].cpu().detach().numpy() * 255
                annotated_img = y_single_rot90[0, 0].cpu().detach().numpy() * 255
                rot90_val_save_img = np.column_stack((seed_img, channels_img, segmented_img, annotated_img))
                # cv2.imwrite("rot90_val_" + str(epoch) + '.png', rot90_val_save_img)

                # Run same sample of x images
                prediction = model(x_anim[:1])  # [N, 2, H, W]
                prediction = F.interpolate(prediction, size=x.shape[2:], mode='nearest')

                # Save image with same x sample
                seed_img = x_anim[0, 0].cpu().detach().numpy() * 255
                channels_img = x_anim[0, 1].cpu().detach().numpy() * 255
                segmented_img = prediction[0, 0].cpu().detach().numpy() * 255
                annotated_img = y_anim[0, 0].cpu().detach().numpy() * 255
                x_anim_save_img = np.column_stack((seed_img, channels_img, segmented_img, annotated_img))
                # cv2.imwrite("x_anim_" + str(epoch) + '.png', x_anim_save_img)

                # # Save if the training score is better
                # if loss_print < best_loss:
                #     best_loss = loss_print
                #     torch.save(model.state_dict(), f'model_{str(epoch)}.model')
                #     cv2.imwrite("train_" + str(epoch) + '.png', train_save_img)
                #     cv2.imwrite("val_" + str(epoch) + '.png', val_save_img)
                #     cv2.imwrite("rot90_val_" + str(epoch) + '.png', rot90_val_save_img)
