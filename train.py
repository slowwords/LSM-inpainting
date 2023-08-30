#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/26 22:17
# @Author: WangZhiwen

import random
import math
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from datasets.dataset import ImageDataset, MaskDataset
from models.networks.generator import Generator
from models.networks.discriminator import Discriminator
from options.train_options import TrainOptions
from torchsummary import summary
from criteria.loss import VGG19
from util.base import sample_data, requires_grad, get_mask, compute_loss, comput_discriminator_loss, data_sampler
from tensorboardX import SummaryWriter
from evals.base import Fid, InceptionV3FeatureExtractor
from facenet_pytorch import MTCNN, InceptionResnetV1
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def trans_range(x):
    return (x + 1) / 2

def train_one_step(image_gt, gen_in, generator, discriminator, g_optimizer, d_optimizer, vgg, inV3, FID, FaceNet, LPIPS, i, masks_data_loader=None):
    if args.mask_mode == "comod_mask":
        mask = get_mask(args)
    elif args.mask_mode == "user_mask":  # mask is provided by user
        mask = next(masks_data_loader)
        mask = 1 - mask
    image_gt = image_gt.to(device)
    mask = mask.to(device)
    image_in = (image_gt * mask).to(device)
    # generator
    requires_grad(generator, True)
    requires_grad(discriminator, False)
    image_comp, image_out, _ = generator(image_in, mask, [gen_in])

    # compute loss
    g_optimizer.zero_grad(set_to_none=True)
    G_losses, G_loss = compute_loss(args, discriminator, vgg, FaceNet, image_out, image_comp, image_gt, mask)
    G_loss.backward()
    g_optimizer.step()

    # discriminator
    D_losses = {}
    requires_grad(generator, False)
    requires_grad(discriminator, True)
    d_optimizer.zero_grad(set_to_none=True)
    D_losses['d_fake'] = comput_discriminator_loss(args, discriminator, image_gt=None, image_comp=image_comp,
                                            mask=mask)
    D_losses['d_real'] = comput_discriminator_loss(args, discriminator, image_gt=image_gt, image_comp=None,
                                            mask=mask)
    D_losses['d_fake'].backward()
    D_losses['d_real'].backward()
    d_optimizer.step()

    # eval
    with torch.no_grad():
        evalutions = {}
        fea_gt = inV3.extract_features(image_gt)
        fea_out = inV3.extract_features(image_out)
        evalutions['fid'] = FID(fea_gt, fea_out).sum().item()
        evalutions['lpips'] = LPIPS(trans_range(image_gt), trans_range(image_out)).mean().item()
        gt = trans_range(image_gt).cpu().permute(0, 2, 3, 1).numpy()[0, :, :, :]
        out = trans_range(image_out).cpu().permute(0, 2, 3, 1).numpy()[0, :, :, :]
        evalutions['ssim'] = ssim(gt, out, multichannel=True)
        evalutions['psnr'] = psnr(gt, out)
    return G_losses, D_losses, evalutions, (image_in, image_comp, image_gt)

def val_one_step(image_in, mask_in, gen_in, generator):
    pass

def save_model(args, generator, discriminator, epoch, both=True):
    save_path = f"{args.ckpt_path}/{args.datasets_name}-{args.crop_size}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if both:
        torch.save(
            {
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict()
            },
            f"{args.ckpt_path}/{args.datasets_name}-{args.crop_size}/model-epoch-{epoch}.pth"
        )
    else:
        torch.save(
            generator.state_dict(),
            f"{args.ckpt_path}/{args.datasets_name}-{args.crop_size}/G-epoch-{epoch}.pth"
        )

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

def save_sample(args, images, epoch, i, mode="train"):
    assert (isinstance(images, list) or isinstance(images, tuple))
    save_path = f"{args.sample_path}/{args.datasets_name}-{args.crop_size}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.batch_size > 12:
        nrow = args.batch_size // 2
    else:
        nrow = args.batch_size
    utils.save_image(
        torch.cat(images, 0),
        f"{args.sample_path}/{args.datasets_name}-{args.crop_size}//{mode}_epoch-{epoch}-sample-{str(i).zfill(6)}.png",
        nrow=nrow,
        normalize=True,
        range=(-1, 1)
    )

def show_loss(pbar, G_losses, D_losses, evalutions, epoch, mode="train"):
    if mode == "train":
        state_msg = (
            f"train epoch[{epoch}/{args.train_epochs}] G_gan: {G_losses['gan']:.3f}, G_vgg: {G_losses['vgg']:.3f}, "
            f"G_l1: {G_losses['l1']:.3f}, G_id: {G_losses['id']:.3f}, "  
            f"D_real: {D_losses['d_real']:.3f}, D_fake: {D_losses['d_fake']:.3f}, "
            f"fid: {evalutions['fid']:.3f}, lpips: {evalutions['lpips']:.3f}, ssim: {evalutions['ssim']:.3f}, psnr: {evalutions['psnr']:.3f}"
        )
        pbar.set_description(state_msg)
    elif mode == "valid":
        state_msg = (
            f"valid epoch[{epoch}/{args.train_epochs}] G_gan: {G_losses['gan']:.3f}, G_vgg: {G_losses['vgg']:.3f}, "
            f"G_l1: {G_losses['l1']:.3f}, G_id: {G_losses['id']:.3f}, " 
            f"D_real: {D_losses['d_real']:.3f}, D_fake: {D_losses['d_fake']:.3f}, "
            f"fid: {evalutions['fid']:.3f}, lpips: {evalutions['lpips']:.3f}, ssim: {evalutions['ssim']:.3f}, psnr: {evalutions['psnr']:.3f}"
        )
        pbar.set_description(state_msg)

def log_writer(args):
    log_path = f"{args.log_path}/{args.datasets_name}-{args.crop_size}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    return writer

def write_loss(writer, G_losses, D_losses, evalutions, epoch, i):
    idx = epoch * i
    writer.add_scalar("G_gan", G_losses['gan'], idx)
    writer.add_scalar("G_vgg", G_losses['vgg'], idx)
    writer.add_scalar("G_l1", G_losses['l1'], idx)
    # writer.add_scalar("G_id", G_losses['id'], idx)
    writer.add_scalar("D_real", D_losses['d_real'], idx)
    writer.add_scalar("D_fake", D_losses['d_fake'], idx)
    writer.add_scalar("fid", evalutions['fid'], idx)
    writer.add_scalar("lpips", evalutions['lpips'], idx)
    writer.add_scalar("ssim", evalutions['ssim'], idx)
    writer.add_scalar("psnr", evalutions['psnr'], idx)


def train(args, image_data_loader, valid_data_loader, generator, discriminator, g_optimizer, d_optimizer, mask_data_loader=None):
    assert args.device is not None
    device = torch.device(args.device)

    # data loader
    loader = sample_data(image_data_loader)
    data_loader = iter(loader)

    # valid loader
    valid_loader = sample_data(valid_data_loader)
    val_data_loader = iter(valid_loader)

    # mask loader
    if args.mask_mode == "user_mask":
        mask_loader = sample_data(mask_data_loader)
        masks_data_loader = iter(mask_loader)
    else:
        masks_data_loader = None

    # models
    vgg = VGG19().to(device)
    inV3 = InceptionV3FeatureExtractor().to(device)
    FID = Fid()
    FaceNet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').eval().to(device)
    for name, parameter in loss_fn_vgg.named_parameters():
        parameter.requires_grad = False

    # log
    writer = log_writer(args)

    for epoch in range(args.train_epochs):
        epoch = epoch + args.start_epoch
        train_len = len(image_data_loader)
        pbar = tqdm(range(train_len))
        for i in pbar:
            generator.train()
            # get input
            try:
                image_gt = next(data_loader)  # real_image.shape=(batch,3,w,h)
            except (OSError, StopIteration):
                data_loader = iter(loader)
                image_gt = next(data_loader)
            # generate noise
            gen_in = torch.randn(
                1, args.batch_size, args.dlatent_size, device=args.device
            ).squeeze(0)
            G_losses, D_losses, evalutions, images = train_one_step(image_gt, gen_in, generator, discriminator,
                                                                    g_optimizer, d_optimizer,
                                                                    vgg, inV3, FID, FaceNet, loss_fn_vgg, i,
                                                                    masks_data_loader)
            show_loss(pbar, G_losses, D_losses, evalutions, epoch)
            write_loss(writer, G_losses, D_losses, evalutions, epoch, i+1)
            if (i + 1) % args.save_sample == 0 or i == 0:
                save_sample(args, images, epoch, i+1)

        if epoch % args.save_interval == 0:
            save_model(args, generator, discriminator, epoch)

if __name__ == "__main__":

    args = TrainOptions().parse

    assert args.device is not None
    device = torch.device(args.device)

    generator = Generator(args).to(device)
    # generator = torch.nn.DataParallel(generator, device_ids=[0, 1])
    print("create step-mod generator.")
    discriminator = Discriminator(args).to(device)
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1])
    print("create step-mod discriminator.")

    g_optimizer = optim.Adam(
        generator.parameters(), lr=args.gen_lr, betas=(0.9, 0.99)
    )

    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.dis_lr, betas=(0.9, 0.99))
    print("create adam optimizer.")

    if args.pre_trained:
        load_model(args, generator, discriminator)

    dataset = ImageDataset(args.image_path, load_size=args.load_size, data_mode=args.data_mode, p=args.p_flip)
    image_data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(
            dataset, shuffle=True
        ),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataset = ImageDataset(args.valid_path, load_size=args.load_size, mode="valid", data_mode=args.data_mode)
    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(
            val_dataset, shuffle=True
        ),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.mask_mode == "comod_mask":
        train(args, image_data_loader, valid_data_loader, generator, discriminator, g_optimizer, d_optimizer)
    elif args.mask_mode == "user_mask":
        mask_dataset = MaskDataset(args.mask_path, load_size=args.load_size)
        mask_data_loader = DataLoader(
            mask_dataset,
            batch_size=args.batch_size,
            sampler=data_sampler(
                mask_dataset, shuffle=False
            ),
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train(args, image_data_loader, valid_data_loader, generator, discriminator, g_optimizer, d_optimizer, mask_data_loader)
