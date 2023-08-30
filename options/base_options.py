#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/26 19:38
# @Author: WangZhiwen

import argparse
import pprint
from collections import OrderedDict

class BaseOptions:

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--in_channels", type=int, default=3,
                            help="in channels")
        self.parser.add_argument("--out_channels", type=int, default=3)
        self.parser.add_argument("--crop_size", type=int, default=512,
                            help="resolution of completed images")
        self.parser.add_argument("--load_size", type=int, default=512,
                            help="image size for training")
        self.parser.add_argument("--device", type=str, default="cuda",
                            help="device for training")
        self.parser.add_argument("--style_mod", type=bool, default=True,
                            help="style mod")
        self.parser.add_argument("--dlatent_size", type=int, default=256,
                            help="d latent size")
        self.parser.add_argument("--cond_mod", type=bool, default=True,
                            help="")
        self.parser.add_argument("--step_mod", type=bool, default=True,
                            help="")
        self.parser.add_argument("--noise_injection", type=bool, default=True)
        self.parser.add_argument("--c_dim", type=int, default=128)

        self.args = self.parser.parse_args()

    @property
    def parse(self):

        # args_dict = OrderedDict(vars(self.args))
        # pprint.pprint(args_dict)

        return self.args