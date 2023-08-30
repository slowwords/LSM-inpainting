#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/27 13:32
# @Author: WangZhiwen

import argparse
import pprint
from collections import OrderedDict
from options.base_options import BaseOptions

class TestOptions(BaseOptions):

    def __init__(self):

        super(TestOptions, self).__init__()

        self.parser.add_argument("--image_path", type=str, default="",
                            help="image path for testing")
        self.parser.add_argument("--mask_path", type=str, default="",
                            help="mask path for testing")
        self.parser.add_argument("--result_path", type=str, default="./results",
                            help="result path for testing")
        self.parser.add_argument("--result_nums", type=int, default=1,
                            help="nums for testing")
        self.parser.add_argument("--mask_mode", type=str, default="comod_mask",
                            help="comod_mask, user_mask")
        self.parser.add_argument("--datasets_name", type=str, default="paris_street_view",
                            help="celeba, paris_street_view, place2")
        self.parser.add_argument("--batch_size", type=int, default=4,
                            help="batch size")
        self.parser.add_argument("--test_iters", type=int, default=10,
                            help="iters for testing")
        self.parser.add_argument("--dropout_rate", type=float, default=0.5,
                            help="ratio of drop out")
        self.parser.add_argument("--pre_trained", type=str, default="",
                            help="pre_trained models")

        self.args = self.parser.parse_args()

    @property
    def parse(self):

        args_dict = OrderedDict(vars(self.args))
        pprint.pprint(args_dict)

        return self.args