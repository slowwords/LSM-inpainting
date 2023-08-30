#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/26 19:32
# @Author: WangZhiwen

import argparse
import pprint
from collections import OrderedDict
from options.base_options import BaseOptions

class TrainOptions(BaseOptions):

    def __init__(self):

        super(TrainOptions, self).__init__()

        self.parser.add_argument("--image_path", type=str, default="",
                            help="image path for training")
        self.parser.add_argument("--valid_path", type=str, default="")
        self.parser.add_argument("--mask_path", type=str, default="",
                            help="mask path for training")
        self.parser.add_argument("--mask_mode", type=str, default="comod_mask",
                            help="comod_mask, user_mask")
        self.parser.add_argument("--data_mode", type=str, default="centercrop",
                            help="centercrop or resize")
        self.parser.add_argument("--p_flip", type=float, default=0.5,
                            help="probability to flip")
        self.parser.add_argument("--ckpt_path", type=str, default="./checkpoint",
                            help="path to save checkpoints")
        self.parser.add_argument("--pre_trained", type=str, default="")
        self.parser.add_argument("--sample_path", type=str, default="./sample",
                            help="path to save sample during training")
        self.parser.add_argument("--datasets_name", type=str, default="celeba",
                            help="celeba, paris_street_view, place2")
        self.parser.add_argument("--batch_size", type=int, default=1,
                            help="batch size")
        self.parser.add_argument("--num_workers", type=int, default=4)
        self.parser.add_argument("--train_epochs", type=int, default=100,
                            help="epochs for training")
        self.parser.add_argument("--start_epoch", type=int, default=1)
        self.parser.add_argument("--save_interval", type=int, default=1,
                            help="epochs to save checkpoints")
        self.parser.add_argument("--save_sample", type=int, default=1000,
                            help="iters to save sample")
        self.parser.add_argument("--epochs_valid", type=int, default=1)
        self.parser.add_argument("--gen_lr", type=float, default=0.002,
                            help="learning rate for generator")
        self.parser.add_argument("--dis_lr", type=float, default=0.002,
                            help="learning rate for discriminator")
        self.parser.add_argument("--gan_mode", type=str, default="softplus",
                            help="gan loss mode,[ls, original, w, hinge, softplus]")
        self.parser.add_argument("--ratio_l1", type=float, default=1.,
                            help="ratio of l1 loss")
        self.parser.add_argument("--ratio_vgg", type=float, default=5.,
                            help="ratio of style loss")
        self.parser.add_argument("--ratio_gan", type=float, default=2.,
                            help="ratio of adversarial loss")
        self.parser.add_argument("--ratio_id", type=float, default=0.1,
                            help="ratio of adversarial loss")
        self.parser.add_argument("--n_critic", type=int, default=1,
                            help="every n iters train discriminator once")
        self.parser.add_argument("--dropout_rate", type=float, default=0.5,
                            help="ratio of drop out")
        self.parser.add_argument("--log_path", type=str, default="./logs")

        self.args = self.parser.parse_args()

    @property
    def parse(self):

        args_dict = OrderedDict(vars(self.args))
        pprint.pprint(args_dict)

        return self.args