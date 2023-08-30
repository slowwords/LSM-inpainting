#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/4/10 17:35
# @Author: WangZhiwen

import torch
from evalution.ssim import SSIM
from evalution.psnr import PSNR
from evalution.lpips_model import Lpips
from evalution.fid_pids_uids import compute_fid_pids_uids
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def eval_compute(args, lpips_model, image_gt, image_comp):

    model_SSIM = SSIM(window_size=args.batch_size)
    ssim = model_SSIM(image_gt*2-1, image_comp*2-1).detach().item()
    psnr = PSNR(image_gt, image_comp)
    lpips = Lpips(lpips_model, image_gt, image_comp)
    fid, pids, uids = compute_fid_pids_uids(image_gt, image_comp, args.device)

    return ssim, psnr, lpips, fid, pids, uids

