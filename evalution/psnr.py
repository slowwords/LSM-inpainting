#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/4/10 16:42
# @Author: WangZhiwen

from skimage.metrics import peak_signal_noise_ratio
import torch

def PSNR(image_gt, image_comp):
    image_gt = image_gt.permute(0, 3, 1, 2).cpu().numpy()
    image_comp = image_comp.permute(0, 3, 1, 2).cpu().numpy()
    return peak_signal_noise_ratio(image_gt, image_comp, data_range=2)


if __name__ == "__main__":
    import cv2
    gt = cv2.imread('D:\data\celeba\gt/00009.png', cv2.IMREAD_COLOR)
    out = cv2.imread("D:\data\celeba/result\DeepFillv2/00009.png", cv2.IMREAD_COLOR)

    gt = (torch.from_numpy(gt).unsqueeze(dim=0).permute(0, 3, 1, 2) - 1) / 127.5
    out = (torch.from_numpy(out).unsqueeze(dim=0).permute(0, 3, 1, 2) - 1) / 127.5
    p = PSNR(gt, out)
    from IPython import embed
    embed()