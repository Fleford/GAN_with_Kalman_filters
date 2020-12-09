import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt
from matplotlib import pyplot as plt


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None,
                 pixel_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))
        convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='nearest')


class Generator(nn.Module):
    def __init__(self, input_code_dim=128, in_channel=128, pixel_norm=True, tanh=True, n_cond_lyrs=4):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        # self.input_layer = nn.Sequential(
        #     EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
        #     PixelNorm(),
        #     nn.LeakyReLU(0.1))

        self.input_layer = nn.Sequential(
            nn.Linear(input_code_dim, 2 * 2 * in_channel)
        )

        self.progression_2 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_4 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_8 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel + n_cond_lyrs, in_channel, 3, 1, pixel_norm=pixel_norm)

        self.conv_cond_2 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)
        self.conv_cond_4 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)
        self.conv_cond_8 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)
        self.conv_cond_16 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)
        self.conv_cond_32 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)
        self.conv_cond_64 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)
        self.conv_cond_128 = ConvBlock(1, n_cond_lyrs, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_2 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)
        self.to_rgb_4 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)
        self.to_rgb_8 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)
        self.to_rgb_16 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)
        self.to_rgb_32 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)
        self.to_rgb_64 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)
        self.to_rgb_128 = EqualConv2d(in_channel + n_cond_lyrs, 1, 1)

        self.max_step = 7

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='nearest')
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, raw_cond_128, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        # Prepare downsampler
        avg_downsampler = torch.nn.AvgPool2d((2, 2), stride=(2, 2))

        # Downsample the conditioning array
        raw_cond_64 = avg_downsampler(raw_cond_128)
        raw_cond_32 = avg_downsampler(raw_cond_64)
        raw_cond_16 = avg_downsampler(raw_cond_32)
        raw_cond_8 = avg_downsampler(raw_cond_16)
        raw_cond_4 = avg_downsampler(raw_cond_8)
        raw_cond_2 = avg_downsampler(raw_cond_4)

        # out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_2 = self.input_layer(input)
        out_2 = out_2.reshape(-1, self.in_channel, 2, 2)

        # raw_cond_2 = F.interpolate(condition_128, size=(2, 2), mode='bilinear', align_corners=False)
        cond_2 = self.conv_cond_2(raw_cond_2)
        out_2 = torch.cat([out_2, cond_2], dim=1)
        out_4 = self.progress(out_2, self.progression_4)
        # out_2 = self.progression_2(out_2)

        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_2(out_2))
            return self.to_rgb_2(out_2)

        # raw_cond_4 = F.interpolate(condition_128, size=(4, 4), mode='bilinear', align_corners=False)
        cond_4 = self.conv_cond_4(raw_cond_4)
        out_4 = torch.cat([out_4, cond_4], dim=1)
        out_8 = self.progress(out_4, self.progression_8)
        # out_4 = self.progression_4(out_4)

        if step == 2:
            return self.output(out_2, out_4, self.to_rgb_2, self.to_rgb_4, alpha)

        # out_4 = torch.cat([out_4, cond_4], dim=1)
        # out_8 = self.progress(out_4, self.progression_8)
        # out_8 = self.input_layer(input)
        # out_8 = out_8.reshape(-1, self.in_channel, 8, 8)

        # raw_cond_8 = F.interpolate(condition_128, size=(8, 8), mode='bilinear', align_corners=False)
        cond_8 = self.conv_cond_8(raw_cond_8)
        out_8 = torch.cat([out_8, cond_8], dim=1)
        out_16 = self.progress(out_8, self.progression_16)

        if step == 3:
            return self.output(out_4, out_8, self.to_rgb_4, self.to_rgb_8, alpha)

        # raw_cond_16 = F.interpolate(condition_128, size=(16, 16), mode='bilinear', align_corners=False)
        cond_16 = self.conv_cond_16(raw_cond_16)
        out_16 = torch.cat([out_16, cond_16], dim=1)
        out_32 = self.progress(out_16, self.progression_32)

        if step == 4:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        # raw_cond_32 = F.interpolate(condition_128, size=(32, 32), mode='bilinear', align_corners=False)
        cond_32 = self.conv_cond_32(raw_cond_32)
        out_32 = torch.cat([out_32, cond_32], dim=1)
        out_64 = self.progress(out_32, self.progression_64)

        if step == 5:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)

        # raw_cond_64 = F.interpolate(condition_128, size=(64, 64), mode='bilinear', align_corners=False)
        cond_64 = self.conv_cond_64(raw_cond_64)
        out_64 = torch.cat([out_64, cond_64], dim=1)
        out_128 = self.progress(out_64, self.progression_128)

        if step == 6:
            return self.output(out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha)

        # raw_cond_128 = F.interpolate(condition_128, size=(128, 128), mode='bilinear', align_corners=False)
        cond_128 = self.conv_cond_128(raw_cond_128)
        out_128 = torch.cat([out_128, cond_128], dim=1)

        if step == 7:
            return self.output(out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha)


class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(feat_dim // 4, feat_dim // 4, 3, 1),
                                          ConvBlock(feat_dim // 4, feat_dim // 2, 3, 1),
                                          ConvBlock(feat_dim // 2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 1, 0)])   # ... 1, 2/2, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(1, feat_dim // 4, 1),
                                       EqualConv2d(1, feat_dim // 4, 1),
                                       EqualConv2d(1, feat_dim // 2, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 1, 1)  # ...out.size(0), 1, 2/2, 2/2)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


# For testing purposes only
if __name__ == "__main__":
    # # # Test code for generator
    cond_array = torch.zeros(4, 1, 128, 128, device='cuda:0') * 2 - 1
    gen_z = torch.randn(4, 128).to('cuda:0')
    generator = Generator(in_channel=128, input_code_dim=128, pixel_norm=False, tanh=False).to('cuda:0')
    discriminator = Discriminator(feat_dim=128).to('cuda:0')
    fake_image = generator(gen_z, cond_array, step=7, alpha=0.02)
    d_result = discriminator(fake_image, step=7, alpha=0.02)
    breakpoint()
