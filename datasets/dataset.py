#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/27 14:12
# @Author: WangZhiwen

from io import BytesIO
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import random

def image_transforms(load_size, mode="resize", p=0.5):
    if mode == "resize":
        return transforms.Compose([
            transforms.Resize([load_size, load_size]),
            transforms.RandomHorizontalFlip(p=p),  # 以0.5的概率水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif mode == "centercrop":
        return transforms.Compose([
            transforms.CenterCrop(size=(load_size, load_size)),
            transforms.RandomHorizontalFlip(p=p),  # 以0.5的概率水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def image_transforms_valid(load_size, mode="resize"):   # Resize or CenterCrop
    if mode == "resize":
        return transforms.Compose([
            transforms.Resize([load_size, load_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif mode == "centercrop":
        return transforms.Compose([
            transforms.CenterCrop(size=(load_size, load_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def mask_transforms(load_size):

    return transforms.Compose([
        transforms.Resize([load_size, load_size]),
        transforms.ToTensor()
    ])

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):

    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    images = sorted(images)

    return images[:min(max_dataset_size, len(images))]

class ImageDataset(Dataset):

    def __init__(self, image_root, load_size, sigma=2., mode="train", data_mode="centercrop", p=0.5):
        super(ImageDataset, self).__init__()

        self.image_files = make_dataset(dir=image_root)

        self.number_image = len(self.image_files)

        self.sigma = sigma

        self.load_size = load_size
        if mode == "train":
            self.image_files_transforms = image_transforms(load_size, mode=data_mode, p=p)
        else:
            self.image_files_transforms = image_transforms_valid(load_size, mode=data_mode)


    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        image = self.image_files_transforms(image.convert('RGB'))

        return image

    def __len__(self):

        return self.number_image

class MaskDataset(Dataset):

    def __init__(self, image_root, load_size, sigma=2.):
        super(MaskDataset, self).__init__()

        self.mask_files = make_dataset(dir=image_root)

        self.number_mask = len(self.mask_files)

        self.sigma = sigma

        self.load_size = load_size

        self.mask_files_transforms = mask_transforms(load_size)

    def __getitem__(self, index):

        mask = Image.open(self.mask_files[index % self.number_mask])
        mask = self.mask_files_transforms(mask).convert('L')

        return mask

    def __len__(self):

        return self.number_mask