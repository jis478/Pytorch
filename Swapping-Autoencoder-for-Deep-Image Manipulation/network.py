# REFELCTION PADDING FOR ResBlock, NO PADDING FOR CONV CHECK

import math
import random
import functools
import operator

import torch
from torch import nn
from layers import *
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

# no padding for conv
# ref padding for ResBlock

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.conv1 = ConvLayer(3, self.in_channels, 1, stride=1)                                                                      # 512x512x32

        self.blocks = []
        for i in range(4):
            self.blocks.append(ResBlock(self.in_channels, self.in_channels*2, padding="reflect"))
            self.in_channels = self.in_channels * 2
        self.blocks = nn.Sequential(*self.blocks)

        self.structure = nn.Sequential(ConvLayer(self.in_channels, self.in_channels, 1, stride=1, padding="valid"),
                                       ConvLayer(512, 8, 1, stride=1, padding="valid")
                                       )

        self.texture = nn.Sequential(ConvLayer(self.in_channels, self.in_channels * 2, 3, stride=2, padding="valid"),
                                     ConvLayer(self.in_channels * 2, self.in_channels * 4, 3, stride=2, padding="valid"),    # 7x7x2048
                                     nn.AdaptiveAvgPool2d(1),                                                              # 1x1x2048
#                                      nn.Flatten(),                                                                         # 2048
#                                      EqualLinear(2048, 2048)                                                               # 2048
                                     ConvLayer(2048, 2048, 1, stride=1, padding="valid")
                                     )

    def forward(self, input):
        # print('encoder start')
        out = self.conv1(input)
        out = self.blocks(out)
        # print('fn_out', out.size())
        s_code = self.structure(out)
        # print('s_code', s_code.size())
        t_code = self.texture(out)
        t_code = torch.flatten(t_code, 1)
        # print('t_code', t_code.size())
        # print('encoder finish')
        return s_code, t_code

import torch.nn.functional as nnf


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.feature_extractor_in_ch = in_ch
        self.feature_extractor_ch = [64, 128, 256, 384, 768, 384]
        # self.feature_extractor_ch = [64, 128, 256, 384, 384, 768, 384]
        # self.classifier_ch = [2048, 2048, 1024, 1]
        self.classifier_ch = [1024, 1024, 512, 1]

        self.conv1 = ConvLayer(3, self.feature_extractor_in_ch, 1, padding="zero")   #원래 3                                                                                      # 128x128x32

        feature_extractor = []
        for i in range(len(self.feature_extractor_ch)-2):
            # print('ch', i, self.feature_extractor_in_ch)
            feature_extractor.append(ResBlock(self.feature_extractor_in_ch, self.feature_extractor_ch[i]))
            self.feature_extractor_in_ch = self.feature_extractor_ch[i]
        feature_extractor.append(ResBlock(self.feature_extractor_in_ch, self.feature_extractor_ch[-2], downsample=False))
        #1
        # feature_extractor.append(ConvLayer(self.feature_extractor_ch[-2], self.feature_extractor_ch[-1], 2, padding="valid"))                 #(W−F+2P)/S+1); 4->2
        feature_extractor.append(ConvLayer(self.feature_extractor_ch[-2], self.feature_extractor_ch[-1], 3, padding="valid"))                 #(W−F+2P)/S+1); 4->2
        self.feature_extractor = nn.Sequential(*feature_extractor)

        classifier = []
        activation = "fused_lrelu"

        #2
        self.classifier_in_ch = self.feature_extractor_ch[-1] * 2 * 2 * 2                                                                # 2 x 768 -> 1 x 1536
        # self.classifier_in_ch = 3072                                                               # 2 x 768 -> 1 x 1536

        for ch in self.classifier_ch:
            if ch == 1:
                activation = None
            classifier.append(EqualLinear(self.classifier_in_ch, ch, activation=activation))
            self.classifier_in_ch = ch
        self.classifier = nn.Sequential(*classifier)

    # 한 개의 patch를 비교할 때 4개의 ref 패치가 필요 함
    def forward(self, ref_patches, patch, ref_num=4):
        # print('patch dis start')
        # print('rf', ref_patches.size())
        patches = self.conv1(ref_patches)
        # print('rf2', patches.size())
        patches = self.feature_extractor(patches)                                               # (b x 4) x 2 x 384
        # print('ps1', patches.size())
        patches = patches.view(-1, ref_num, 384, 2, 2)                                             # b x 4 x 2 x 384
        # print('ps2', patches.size())
        patches_avg = torch.mean(patches, dim=1)                                                # b x 2 x 384
        # print('ps3', patches_avg.size())
        # print('ppp1', patch.size())
        patch = self.conv1(patch)
        # print('ppp2', patch.size())
        patch = self.feature_extractor(patch)                                                   # b x 2 x 384
        # print('pa', patches_avg.size())
        # print('p', patch.size())
        patch_cat = torch.cat((patches_avg, patch), 1)                                          # b x 2 x 768
        # print('patch_cat', patch_cat.size())
        patch_cat = patch_cat.view(patch_cat.size()[0],-1)                                        # b x 1 x 1536
        # print('patch_cat', patch_cat.size())
        out = self.classifier(patch_cat)
        # print('out', out.size())
        # print('patch dis finish')
        return out




class Generator(nn.Module):
    def __init__(
            self,
            style_dim
    ):
        super().__init__()

        self.style_dim = style_dim

        self.ResBlocks = nn.Sequential(
            UpsamplingResBlock(8, 128, self.style_dim, False),
            UpsamplingResBlock(128, 256, self.style_dim, False),
            UpsamplingResBlock(256, 384, self.style_dim, False),
            UpsamplingResBlock(384, 512, self.style_dim, False),
            UpsamplingResBlock(512, 512, self.style_dim, True),
            UpsamplingResBlock(512, 512, self.style_dim, True),
            UpsamplingResBlock(512, 256, self.style_dim, True),
            UpsamplingResBlock(256, 128, self.style_dim, True),                                                          # 512x512x128
        )
        self.conv1 = ConvLayer(128, 3, 1, activate=False)                                                                # 512x512x3 (zero padding:1)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.ResBlocks)

        out = structure
        for ResBlock, noise in zip(self.ResBlocks, noises):
            out = ResBlock(out, texture, noise)

        out = self.conv1(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
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

        self.final_conv = ConvLayer(in_channel, channels[4], 3)

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        # print('discriminator1')
        out = self.convs(input)
        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)
        return out
