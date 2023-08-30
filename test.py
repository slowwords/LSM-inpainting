#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/27 21:55
# @Author: WangZhiwen

import os
import torch
from tqdm import tqdm
from options.test_options import TestOptions
from util.base import sample_data, get_mask, data_sampler
from models.networks.generator import Generator
from torchvision import utils
from torch.utils.data import DataLoader
from datasets.dataset import ImageDataset, MaskDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(args, generator, discriminator, both=True):
    model_dict = torch.load(args.pre_trained)
    # generator
    G_dict = generator.state_dict()
    G_pre_dict = model_dict['generator']
    G_pred_dict = {k: v for k, v in G_pre_dict.items() if k in G_dict}
    G_dict.update(G_pred_dict)
    generator.load_state_dict(G_dict, strict=False)
    print(f"load pretrained G weights")
    if both:
        # discriminator
        D_dict = discriminator.state_dict()
        D_pre_dict = model_dict['discriminator']
        D_pred_dict = {k: v for k, v in D_pre_dict.items() if k in D_dict}
        D_dict.update(D_pred_dict)
        discriminator.load_state_dict(D_dict, strict=False)
        print(f"load pretrained D weights")

def save_result(args, image, i, mode="gt"):

    save_path = f"{args.result_path}/{args.datasets_name}-{args.crop_size}/{mode}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    utils.save_image(
        image,
        f"{args.result_path}/{args.datasets_name}-{args.crop_size}/{mode}/result-{str(i).zfill(6)}_.png",
        nrow=1,
        normalize=True,
        range=(-1, 1)
    )

def test(args, generator, image_data_loader, mask_data_loader=None):

    assert args.device is not None
    device = torch.device(args.device)

    # data loader
    loader = sample_data(image_data_loader)
    data_loader = iter(loader)
    if args.mask_mode == "user_mask":
        mask_loader = sample_data(mask_data_loader)
        data_mask_loader = iter(mask_loader)

    pbar = tqdm(range(args.test_iters))
    with torch.no_grad():
        generator.eval()
        for i in pbar:
            # get input
            try:
                image_gt = next(data_loader)  # real_image.shape=(batch,3,w,h)
            except (OSError, StopIteration):
                data_loader = iter(loader)
                image_gt = next(data_loader)
            if args.mask_mode == "comod_mask":
                mask = get_mask(args)
            elif args.mask_mode == "user_mask":
                mask = next(data_mask_loader)
                mask = 1 - mask

            image_gt = image_gt.to(device)
            mask = mask.to(device)
            image_masked = (image_gt * mask).to(device)

            gen_in = torch.randn(
                1, args.batch_size, args.dlatent_size, device=args.device
            ).squeeze(0)
            image_comp, image_out, _ = generator(image_masked, mask, [gen_in])

            save_result(args, image_masked, i, mode="in")
            save_result(args, 1-mask, i, mode="mask")
            save_result(args, image_comp, i, mode="out")
            save_result(args, image_gt, i, mode="gt")

if __name__ == "__main__":

    args = TestOptions().parse

    assert args.device is not None
    device = torch.device(args.device)

    generator = Generator(args).to(device)

    if args.pre_trained != '':
        generator.load_state_dict(torch.load(args.pre_trained)['generator'])
        print("Success for loading pre-trained model!")
    else:
        print('Please provide pre-trained model!')

    dataset = ImageDataset(args.image_path, load_size=args.load_size)
    image_data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(
            dataset, shuffle=False
        ),
        drop_last=True
    )
    if args.mask_mode == "comod_mask":
        test(args, generator, image_data_loader)
    elif args.mask_mode == "user_mask":
        mask_dataset = MaskDataset(args.mask_path, load_size=args.load_size)
        mask_data_loader = DataLoader(
            mask_dataset,
            batch_size=args.batch_size,
            sampler=data_sampler(
                mask_dataset, shuffle=False
            ),
            drop_last=True
        )
        test(args, generator, image_data_loader, mask_data_loader)