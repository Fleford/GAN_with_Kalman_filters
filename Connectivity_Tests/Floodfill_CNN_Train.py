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


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open('training_log.txt', 'a') as file:
        print(*args, **kwargs, file=file)


def floodfill_data_pair(_):
    # Load in training image
    training_img = cv2.imread('ti.png', 0)

    # Randomly pick a part of the training image
    window_size = 256
    top_left_row_coord = np.random.randint(training_img.shape[0] - window_size + 1)
    top_left_col_coord = np.random.randint(training_img.shape[1] - window_size + 1)

    # Extract window
    img_channels = training_img[top_left_row_coord:top_left_row_coord + window_size,
             top_left_col_coord:top_left_col_coord + window_size]

    # Prep test image
    img_channels[img_channels == 255] = 1

    # Scale down the image
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
    for _ in range(2):
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

    sf_shrink = 0.4
    best_val_score = 0
    best_train_val_score = 0

    # Clear log file
    with open('training_log.txt', 'w') as file:
        print()

    # Write header line in log file
    print_log('epoch, loss_print, context_loss_print, error_ratio_print, pos_only_ratio_print')

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

        loss = context_loss - ref_loss

        loss_print = loss.item()
        context_loss_print = context_loss.item()
        print(loss_print, context_loss_print, error_ratio_print)

        loss.backward()
        optim.step()

        if epoch % 100 == 0:
            # torch.save(model.state_dict(), f'model_{str(epoch)}.model')
            with torch.no_grad():
                # Prep saved image for training sample
                seed_img = x[0, 0].cpu().detach().numpy() * 255
                channels_img = x[0, 1].cpu().detach().numpy() * 255
                segmented_img = prediction[0, 0].cpu().detach().numpy() * 255
                annotated_img = y[0, 0].cpu().detach().numpy() * 255
                train_save_img = np.column_stack((seed_img, channels_img, segmented_img, annotated_img))
                cv2.imwrite("train_" + str(epoch) + '.png', train_save_img)

                print_log(epoch, loss_print, context_loss_print,
                          error_ratio_print,
                          pos_only_ratio_print)
