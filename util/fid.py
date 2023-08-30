#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/1/26 19:52
# @Author: WangZhiwen

import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np
from torchvision.models import inception_v3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FID:
    def __init__(self, num_images=2, device='cpu'):
        self.num_images = num_images
        self.device = device

    def get_feature_vector(self, x):
        raise NotImplementedError('Subclass must implement abstract method')

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        cov_mean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(cov_mean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            cov_mean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        fid = np.real(np.trace(sigma1 + sigma2 - 2 * cov_mean))
        mean_diff = np.linalg.norm(mu1 - mu2)
        return fid, mean_diff

    def __call__(self, tensor1, tensor2):
        with torch.no_grad():
            tensor1 = tensor1.to(self.device)
            tensor2 = tensor2.to(self.device)
            assert tensor1.shape == tensor2.shape, \
                f"Two tensors must have the same shape, but got {tensor1.shape} and {tensor2.shape}"
            assert tensor1.ndim == 4 and tensor2.ndim == 4, \
                f"Input tensors must be 4D, but got {tensor1.ndim} and {tensor2.ndim}"

            num_images1, num_images2 = tensor1.shape[0], tensor2.shape[0]
            assert num_images1 >= self.num_images and num_images2 >= self.num_images, \
                f"Number of images must be greater than or equal to {self.num_images}, " \
                f"but got {num_images1} and {num_images2}"

            # Calculate the activations for the real images
            num_batches1 = int(np.ceil(num_images1 / self.num_images))
            features1 = np.zeros((num_images1, 2048), dtype=np.float32)
            for i in range(num_batches1):
                j = min(num_images1, i + self.num_images)
                features1[i:j] = self.get_feature_vector(tensor1[i:j]).cpu().numpy()

            # Calculate the activations for the generated images
            num_batches2 = int(np.ceil(num_images2 / self.num_images))
            features2 = np.zeros((num_images2, 2048), dtype=np.float32)
            for i in range(num_batches2):
                j = min(num_images2, i + self.num_images)
                features2[i:j] = self.get_feature_vector(tensor2[i:j]).cpu().numpy()

            # Calculate the means and covariance matrices of the activations
            mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
            mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

            # Calculate FID
            fid, _ = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            return fid


class FIDInceptionV3(FID):
    def __init__(self, num_images=50000, device='cuda'):
        super().__init__(num_images=num_images, device=device)
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
        self.inception_model.fc = torch.nn.Identity()

    def get_feature_vector(self, x):
        # x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.inception_model(x)
        return x.mean(dim=0, keepdim=True)


if __name__ == "__main__":

    img1 = torch.ones(50, 3, 256, 256)
    img2 = torch.zeros(50, 3, 256, 256)
    fidmodel = FIDInceptionV3(num_images=50, device="cpu")
    fid = fidmodel(img1, img2)
    from IPython import embed
    embed()