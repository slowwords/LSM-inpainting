#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/26 21:12
# @Author: WangZhiwen

from util.create_mask import MaskCreator
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import random
from criteria.loss import VGGLoss, GANLoss, KLDLoss, contrastiveLoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def get_mask(args):
    h, w = args.load_size, args.load_size
    device = torch.device(args.device)
    assert args.mask_mode == "comod_mask"
    mask_creator = MaskCreator().to(device)
    mask_list = []
    for i in range(args.batch_size):
        ri = random.randint(0, 3)
        if ri == 0:
            mask = mask_creator.stroke_mask(h, w, max_length=min(h, w) / 2).astype(np.float32)
            mask = 1 - mask
        elif ri == 1:
            mask = mask_creator.rectangle_mask(h, w, min(h, w) // 4, min(h, w) // 2).astype(np.float32)
        else:
            mask = mask_creator.random_mask(h, w).astype(np.float32)
        mask_list.append([mask])
    masks = torch.from_numpy(np.array(mask_list))
    pi = random.randint(0, 1)
    if pi == 0:
        masks = 1 - masks
    return masks

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def data_sampler(dataset, shuffle):

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get("mult", 1)
        group["lr"] = lr * mult

def compute_loss(args, discriminator, vgg, FaceNet, image_out, image_comp, image_gt, mask, consine=True):
    device = torch.device(args.device)
    if args.device == "cpu":
        FloatTensor = torch.FloatTensor
    else:
        FloatTensor = torch.cuda.FloatTensor

    vgg_loss = VGGLoss()
    gan_loss = GANLoss(gan_mode=args.gan_mode, tensor=FloatTensor)
    pred_fake = discriminator(image_comp, mask)

    G_losses = {}
    G_losses["gan"] = gan_loss(pred_fake, True, for_discriminator=False).mean() * args.ratio_gan

    G_losses['vgg'] = vgg_loss(vgg(image_out), vgg(image_gt)).mean() * args.ratio_vgg
    G_losses["l1"] = torch.nn.functional.l1_loss(image_out, image_gt).mean() * args.ratio_l1
    if consine:
        target = torch.ones(1).to(device)
        G_losses['id'] = torch.nn.functional.cosine_embedding_loss(FaceNet(image_out), FaceNet(image_gt), target=target).mean() *\
                        args.ratio_id
    else:
        G_losses['id'] = torch.nn.functional.l1_loss(FaceNet(image_out), FaceNet(image_gt)).mean() * args.ratio_id

    return G_losses, G_losses["gan"]+G_losses['vgg']+G_losses["l1"] + G_losses['id']

def comput_discriminator_loss(args, discriminator, image_gt, image_comp=None, mask=None):

    assert mask is not None
    assert image_comp is not None or image_gt is not None
    assert image_comp is None or image_gt is None

    if args.device == "cpu":
        FloatTensor = torch.FloatTensor
    else:
        FloatTensor = torch.cuda.FloatTensor

    gan_loss = GANLoss(gan_mode=args.gan_mode, tensor=FloatTensor)

    if image_comp is not None:
        image_comp = image_comp.detach()
        pred_fake = discriminator(image_comp, mask)
        d_fake = gan_loss(pred_fake, False, for_discriminator=True).mean() * args.ratio_gan
        return d_fake
    elif image_gt is not None:
        pred_real = discriminator(image_gt, mask)
        d_real = gan_loss(pred_real, True, for_discriminator=True).mean() * args.ratio_gan
        return d_real

def extract_patches(x, kernel_size=3, stride=1):

    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()
