#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/27 13:32
# @Author: WangZhiwen

import argparse
import pprint
from collections import OrderedDict

class EvalOptions:

    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument("--in_channels", type=int, default=3,
                            help="in channels")
        parser.add_argument("--out_channels", type=int, default=3)
        parser.add_argument("--image_path", type=str, default="F:\dataset\CelebA-HQ-img\img",
                            help="image path for testing")
        parser.add_argument("--mask_path", type=str, default=None,
                            help="mask path for testing")
        parser.add_argument("--result_path", type=str, default="./results",
                            help="result path for testing")
        parser.add_argument("--result_nums", type=int, default=5,
                            help="nums for testing")
        parser.add_argument("--mask_mode", type=str, default="comod_mask",
                            help="comod_mask, user_mask")
        parser.add_argument("--datasets_name", type=str, default="celeba",
                            help="celeba, paris_street_view, place2")
        parser.add_argument("--batch_size", type=int, default=4,
                            help="batch size")
        parser.add_argument("--crop_size", type=int, default=256,
                            help="resolution of completed images")
        parser.add_argument("--load_size", type=int, default=256,
                            help="image size for training")
        parser.add_argument("--eval_iters", type=int, default=10,
                            help="iters for eval")
        parser.add_argument("--device", type=str, default="cuda",
                            help="device for training")
        parser.add_argument("--dropout_rate", type=float, default=0.5,
                            help="ratio of drop out")
        parser.add_argument("--pre_trained", type=str, default="./checkpoint/celeba/resolution-256-train_iters-10.pth",
                            help="pre_trained models")
        parser.add_argument("--style_mod", type=bool, default=True,
                            help="style mod")
        parser.add_argument("--dlatent_size", type=int, default=512,
                            help="d latent size")
        parser.add_argument("--cond_mod", type=bool, default=True,
                            help="")
        parser.add_argument("--step_mod", type=bool, default=True,
                            help="")
        parser.add_argument("--noise_injection", type=bool, default=True)
        self.args = parser.parse_args()

    @property
    def parse(self):

        args_dict = OrderedDict(vars(self.args))
        # pprint.pprint(args_dict)

        return self.args