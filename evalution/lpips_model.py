#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/4/10 17:02
# @Author: WangZhiwen

import lpips
import torch

def Lpips(lpips_net, image_gt, image_out):
    lpips_score = lpips_net(image_gt, image_out).squeeze(-1).squeeze(-1).squeeze(-1)
    return lpips_score.mean().detach().item()

if __name__ == "__main__":
    t = torch.ones(16, 1, 1, 1)
    from IPython import embed
    embed()

