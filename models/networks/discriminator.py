#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/26 20:12
# @Author: WangZhiwen

import torch
import torch.nn as nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.stylegan2.model import ConvLayer, EqualLinear
from collections import OrderedDict

class Discriminator(BaseNetwork):

    def __init__(self, args):

        resolution = args.crop_size
        fmap_base = 16 << 10
        fmap_decay = 1
        fmap_min = 1
        fmap_max = 512
        resample_kernel = [1,3,3,1]
        mbstd_group_size = 4
        mbstd_num_features = 1

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        # Building blocks for main layers.
        super().__init__()
        layers = []
        c_in = args.in_channels + 1
        layers.append(
                (
                    "ToRGB",
                    ConvLayer(
                        c_in,
                        nf(resolution_log2-1),
                        kernel_size=3,
                        activate=True)
                    )
                )

        class Block(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.Conv0 = ConvLayer(
                        nf(res-1),
                        nf(res-1),
                        kernel_size=3,
                        activate=True)
                self.Conv1_down = ConvLayer(
                        nf(res-1),
                        nf(res-2),
                        kernel_size=3,
                        downsample=True,
                        blur_kernel=resample_kernel,
                        activate=True)
                self.Skip = ConvLayer(
                        nf(res-1),
                        nf(res-2),
                        kernel_size=1,
                        downsample=True,
                        blur_kernel=resample_kernel,
                        activate=False,
                        bias=False)

            def forward(self, x):
                t = x
                x = self.Conv0(x)
                x = self.Conv1_down(x)
                t = self.Skip(t)
                x = (x + t) * (1/np.sqrt(2))
                return x
        # Main layers.
        for res in range(resolution_log2, 2, -1):
            layers.append(
                    (
                        '%dx%d' % (2**res, 2**res),
                        Block(res)
                        )
                    )
        self.convs = nn.Sequential(OrderedDict(layers))
        # Final layers.
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features

        self.Conv4x4 = ConvLayer(nf(1)+1, nf(1), kernel_size=3, activate=True)
        self.Dense0 = EqualLinear(nf(1)*4*4, nf(0), activation='fused_lrelu')
        self.Output = EqualLinear(nf(0), 1)

    def forward(self, images_in, masks_in):
        y = torch.cat([masks_in - 0.5, images_in], 1)
        out = self.convs(y)
        batch, channel, height, width = out.shape
        # group_size = min(batch, self.mbstd_group_size)
        group_size = batch
        stddev = out.view(
            group_size,
            -1,
            self.mbstd_num_features,
            channel // self.mbstd_num_features,
            height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group_size, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.Conv4x4(out)
        out = out.view(batch, -1)
        out = self.Dense0(out)
        out = self.Output(out)
        return out

if __name__ == "__main__":
    from options.train_options import TrainOptions
    args = TrainOptions().parse
    net_D = Discriminator(args).cuda()
    image = torch.ones(6, 3, 256, 256).cuda()
    mask = torch.ones(6, 1, 256, 256).cuda()
    fake = net_D(image, mask)
    from IPython import embed
    embed()