from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from progan_modules import Generator, Discriminator
from utils import get_texture2D_iter
from matplotlib import pyplot as plt


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)
        return data_loader

    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size + int(image_size * 0.2) + 1),
        transforms.RandomCrop(image_size),
        # transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    loader = dataloader(transform)

    return loader


def generate_condition(input_matrix, density=10):
    # ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(input_matrix, dtype=torch.float32)
    random_matrix = torch.randint_like(ref_k_array, 2)
    for x in range(density):
        random_matrix = random_matrix * torch.randint_like(ref_k_array, 2)

    # Enlarge condition points
    sf = 2
    avg_downsampler = torch.nn.MaxPool2d((sf, sf), stride=(sf, sf))
    random_matrix = avg_downsampler(random_matrix)
    random_matrix = F.interpolate(random_matrix, scale_factor=sf, mode="nearest")

    output_matrix = ref_k_array * random_matrix
    # output_matrix = torch.zeros_like(input_matrix)
    return output_matrix.cuda(), torch.as_tensor(random_matrix, dtype=torch.float32, device=device)


def train(generator, discriminator, init_step, loader, total_iter=600000, max_step=7):      # max_step=7)
    step = init_step  # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
    # data_loader = sample_data(loader, 4 * 2 ** step)
    # dataset = iter(data_loader)

    # total_iter = 600000
    total_iter_remain = total_iter - (total_iter // max_step) * (step - 1)

    pbar = tqdm(range(total_iter_remain))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    from datetime import datetime
    import os
    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt' % (trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d' % (trial_name, date_time.date(), date_time.hour, date_time.minute)

    os.mkdir(log_folder)
    os.mkdir(log_folder + '/checkpoint')
    os.mkdir(log_folder + '/sample')

    config_file_name = os.path.join(log_folder, 'train_config_' + post_fix)
    config_file = open(config_file_name, 'w')
    config_file.write(str(args))
    config_file.close()

    log_file_name = os.path.join(log_folder, 'train_log_' + post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,cntxt_loss,ds_cntxt_loss\n')
    log_file.close()

    from shutil import copy
    copy('train_with_z_swap_top_bttm.py', log_folder + '/train_%s.py' % post_fix)
    copy('progan_modules.py', log_folder + '/model_%s.py' % post_fix)
    copy('utils.py', log_folder + '/utils_%s.py' % post_fix)

    # alpha = 0
    # one = torch.FloatTensor([1]).to(device)
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    # Prepare reference batch for display
    # data_iter_sample = get_texture2D_iter('ti/', batch_size=5 * 10)
    # real_image_raw_res_sample = torch.Tensor(next(data_iter_sample)).to(device)
    # cond_array_sample, cond_mask_sample = generate_condition(real_image_raw_res_sample)
    # cond_array_sample = torch.zeros(batch_size, 1, 128, 128, device='cuda:0')

    # broadcast first cond_array to whole batch
    # one_cond_array_sample = torch.zeros_like(cond_array_sample)
    # for slice in range(len(cond_array_sample) // 2):
    #     cond_array_sample[slice] = cond_array_sample[0]
    # cond_array_sample = one_cond_array_sample

    data_iter = get_texture2D_iter('ti/', batch_size=batch_size)
    # cntxt_loss = torch.FloatTensor([69]).to(device)

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2 / (total_iter // max_step)) * iteration)

        if iteration > total_iter // max_step:
            alpha = 0
            iteration = 0
            step += 1

            if step > max_step:
                alpha = 1
                step = max_step

        # Scale training image using avg downsampling
        real_image_raw_res = torch.Tensor(next(data_iter)).to(device)
        kernel_width = 2 ** (7 - step)      # 2 ** (max_step - step)
        avg_downsampler = torch.nn.AvgPool2d((kernel_width, kernel_width), stride=(kernel_width, kernel_width))
        cond_downsampler = torch.nn.MaxPool2d((kernel_width, kernel_width), stride=(kernel_width, kernel_width))
        real_image = avg_downsampler(real_image_raw_res)
        # plt.matshow(real_image[0, 0].cpu().detach().numpy())
        # plt.show()

        iteration += 1

        ### 1. train Discriminator
        b_size = real_image.size(0)
        # label = torch.zeros(b_size).to(device)
        real_predict = discriminator(real_image, step=step, alpha=alpha)

        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data: vector for Generator
        # second half of the batch is a top-swapped version of the first half
        gen_z_first_half = torch.randn(b_size // 2, input_z_channels, 2, 2).to(device)
        gen_z_top_swap = torch.flip(gen_z_first_half[:, :, 0:gen_z_first_half.shape[2] // 2, :], dims=[0])
        gen_z_bttm = gen_z_first_half[:, :, gen_z_first_half.shape[2] // 2:gen_z_first_half.shape[2], :]
        gen_z_second_half = torch.cat((gen_z_top_swap, gen_z_bttm), dim=2)
        # gen_z_second_half = torch.cat((gen_z_bttm, gen_z_top_swap), dim=2)

        gen_z = torch.cat((gen_z_first_half, gen_z_second_half), dim=0)

        # generate condition array
        # cond_array, cond_mask = generate_condition(real_image_raw_res)

        # # broadcast first raw image to the whole batch
        # # one_real_image_raw_res = torch.zeros_like(real_image_raw_res)
        # for slice in range(len(real_image_raw_res)//4):
        #     real_image_raw_res[slice] = real_image_raw_res[0]
        # # real_image_raw_res = one_real_image_raw_res
        #
        # # broadcast first cond array to the whole batch
        # # one_cond_array = torch.zeros_like(cond_array)
        # for slice in range(len(cond_array)//4):
        #     cond_array[slice] = cond_array[0]
        # # cond_array = one_cond_array
        #
        # # broadcast first cond mask to the whole batch
        # # one_cond_mask = torch.zeros_like(cond_mask)
        # for slice in range(len(cond_mask)//4):
        #     cond_mask[slice] = cond_mask[0]
        # # cond_mask = one_cond_mask

        fake_image = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward(one)
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()

            predict = discriminator(fake_image, step=step, alpha=alpha)

            ## Calculate context loss (for z swap)
            fake_image_first_half = fake_image[:fake_image.shape[0]//2]
            fake_image_top_swap = torch.flip(fake_image_first_half[:, :,
                                             0:fake_image_first_half.shape[2] // 2, :], dims=[0])
            fake_image_bttm = fake_image_first_half[:, :,
                              fake_image_first_half.shape[2] // 2:fake_image_first_half.shape[2], :]
            fake_image_true_z_swap = torch.cat((fake_image_top_swap, fake_image_bttm), dim=2)
            fake_image_gen_z_swap = fake_image[fake_image.shape[0]//2:]

            cond_mask = torch.zeros_like(fake_image_true_z_swap)
            cond_mask[:, :, 0:cond_mask.shape[2] // 8, :] = 1
            cond_mask[:, :, -cond_mask.shape[2] // 8:, :] = 1
            if fake_image_true_z_swap.shape[2] >= 8:
                context_loss_array = ((fake_image_gen_z_swap - fake_image_true_z_swap) ** 2) * cond_mask
            else:
                context_loss_array = torch.zeros_like(fake_image_true_z_swap)
            # context_loss_value = torch.log(torch.sum(context_loss_array) + 1.0)
            context_loss_value = torch.sum(context_loss_array)

            # Calculate context loss (conditioning hard data)
            # fake_image_upsampled = F.interpolate(fake_image, size=(128, 128), mode="nearest")

            # real_image_upsampled = F.interpolate(real_image, size=(128, 128), mode="nearest")
            # context_loss_array = ((fake_image_upsampled - real_image_upsampled) ** 2) * cond_mask

            # context_loss_array = ((fake_image_upsampled - real_image_raw_res) ** 2) * cond_mask

            # ds_cond_mask = cond_downsampler(cond_mask)
            # ds_context_loss_array = ((fake_image_upsampled - real_image_raw_res) ** 2) * cond_mask
            # ds_context_loss_value = torch.sum(ds_context_loss_array)
            # ds_cntxt_loss = ds_context_loss_value.item()

            # context_loss_value = torch.sum(context_loss_array).log()

            loss = -predict.mean() + 0.02 * context_loss_value
            # loss = -predict.mean()
            gen_loss_val += loss.item()
            cntxt_loss = context_loss_value.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        if (i + 1) % 1000 == 0 or i == 0:
            with torch.no_grad():
                sample_z_first_half = torch.randn(4*10 // 2, input_z_channels, 2, 2).to(device)
                sample_z_top_swap = torch.flip(sample_z_first_half[:, :, 0:sample_z_first_half.shape[2] // 2, :], dims=[0])
                sample_z_bttm = sample_z_first_half[:, :, sample_z_first_half.shape[2] // 2:sample_z_first_half.shape[2], :]
                sample_z_second_half = torch.cat((sample_z_top_swap, sample_z_bttm), dim=2)
                sample_z = torch.cat((sample_z_first_half, sample_z_second_half), dim=0)


                # sample_z = torch.randn(4*10, input_z_channels, 2, 2).to(device)
                images = g_running(sample_z, step=step, alpha=alpha).data.cpu()
                images = F.interpolate(images, size=(128, 128), mode="nearest")
                utils.save_image(
                    images,
                    f'{log_folder}/sample/{str(i + 1).zfill(6)}.png',
                    nrow=10,
                    normalize=True,
                    range=(-1, 1))

        if (i + 1) % 10000 == 0 or i == 0:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_d.model')
            except:
                pass

        if (i + 1) % 500 == 0:
            state_msg = (f'{i + 1}; G: {gen_loss_val / (500 // n_critic):.3f}; D: {disc_loss_val / 500:.3f};'
                         f' Grad: {grad_loss_val / 500:.3f}; Alpha: {alpha:.3f}; Step: {step:.3f}; Iteration: {iteration:.3f};'
                         f' cntxt_loss: {cntxt_loss:.3f};')
            print(real_image.shape)

            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n" % (
            gen_loss_val / (500 // n_critic), disc_loss_val / 500)
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            print(state_msg)
            # pbar.set_description(state_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--path', type=str, default="all_images/",
                        help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="test18", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=6,
                        help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=128,
                        help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=32, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train D how many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=1,
                        help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=400000,
                        help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true",
                        help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true",
                        help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')

    args = parser.parse_args()

    print(str(args))

    trial_name = args.trial_name
    device = torch.device("cuda:%d" % (args.gpu_id))
    input_z_channels = args.z_dim    # Thicness of z vector
    batch_size = args.batch_size
    n_critic = args.n_critic

    generator = Generator(in_channel=args.channel, input_z_channels=input_z_channels, pixel_norm=args.pixel_norm,
                          tanh=args.tanh).to(device)
    discriminator = Discriminator(feat_dim=args.channel).to(device)
    g_running = Generator(in_channel=args.channel, input_z_channels=input_z_channels, pixel_norm=args.pixel_norm,
                          tanh=args.tanh).to(device)

    # # you can directly load a pretrained model here
    # generator.load_state_dict(torch.load('./tr checkpoint/150000_g.model'))
    # g_running.load_state_dict(torch.load('checkpoint/150000_g.model'))
    # discriminator.load_state_dict(torch.load('checkpoint/150000_d.model'))

    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=(args.lr * 0.08), betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    loader = imagefolder_loader(args.path)

    train(generator, discriminator, args.init_step, loader, args.total_iter)
