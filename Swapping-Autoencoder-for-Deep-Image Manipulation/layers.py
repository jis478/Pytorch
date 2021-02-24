
import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        # print('Blur', out.size())

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            # print('weight', self.weight.size())
            # print('input', input.size())
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)               # 2048 -> in_channel

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        # print('input_shape', batch, in_channel, height, width)
        # print('style', style.size())
        # print('mod_style', self.modulation(style).size())
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=1,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        stride = 1,
        padding="zero",
    ):
        layers = []

        self.padding = 0

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)    # check
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

#
# class ConvLayer(nn.Sequential):
#     def __init__(
#             self,
#             in_channel,
#             out_channel,
#             kernel_size=1,
#             upsample=False,
#             downsample=False,
#             blur_kernel=[1, 3, 3, 1],
#             bias=True,
#             activate=True,
#             stride=1,
#             valid=False
#     ):
#         layers = []
#
#         if downsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) + (kernel_size - 1)
#             pad0 = (p + 1) // 2
#             pad1 = p // 2
#
#             layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
#
#             stride = 2
#             self.padding = 0
#
#         if upsample:
#             layers.append(
#                 EqualConvTranspose2d(
#                     in_channel,
#                     out_channel,
#                     kernel_size,
#                     padding=0,
#                     stride=2,
#                     bias=bias and not activate,
#                 )
#             )
#
#             factor = 2
#             p = (len(blur_kernel) - factor) - (kernel_size - 1)
#             pad0 = (p + 1) // 2 + factor - 1
#             pad1 = p // 2 + 1
#
#             layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
#
#         else:
#             if valid:
#                 self.padding = 0
#             else:
#                 self.padding = kernel_size // 2
#
#         layers.append(
#             EqualConv2d(
#                 in_channel,
#                 out_channel,
#                 kernel_size,
#                 padding=self.padding,
#                 stride=stride,
#                 bias=bias and not activate
#             )
#         )
#
#         if activate:
#             layers.append(FusedLeakyReLU(out_channel, bias=bias))
#
#         super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True, padding="zero"):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)
        self.conv2 = ConvLayer(out_channel, out_channel, 3, downsample=downsample, padding=padding)

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
            )
        else:
            self.skip = None

    def forward(self, input):
        # print('res input', input.size())
        out = self.conv1(input)
        # print('b', out.size())
        out = self.conv2(out)
        # print('res output1', out.size())
        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        # print('out', out.size())
        # print('skip', skip.size())
        out = (out + skip) / math.sqrt(2)
        # print('res output2', skip.size())
        return out




class UpsamplingResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample):
        super().__init__()

        self.upsample = upsample
        self.conv1 = StyledConv(in_channel,
                                out_channel,
                                3,
                                style_dim,
                                upsample=upsample
                                )

        self.conv2 = StyledConv(out_channel,
                                out_channel,
                                3,
                                style_dim
                                )

        if self.upsample or in_channel != out_channel:
            self.skip = ConvLayer(
                        in_channel,
                        out_channel,
                        1,
                        upsample=upsample,
                        bias=False,
                        activate=False,
                    )
        else:
            self.skip = None

    def forward(self, input, texture, noise):
        out = self.conv1(input, texture, noise)
        out = self.conv2(out, texture, noise)
        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input
        out = (out + skip) / math.sqrt(2)
        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

