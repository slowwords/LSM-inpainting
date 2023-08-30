#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/2/13 14:51
# @Author: WangZhiwen

import torch
import torch.nn as nn
from torchvision import models

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, images):
        self.model.eval()
        features = self.model(images)
        return features

class Fid(nn.Module):
    def __init__(self):
        super(Fid, self).__init__()

    def torch_cov(self, input_vec: torch.tensor):
        mean = torch.mean(input_vec, dim=0).unsqueeze(dim=0)
        x = input_vec - mean
        cov_matrix = torch.matmul(x.T, x) / (x.shape[1] - 1)
        return cov_matrix

    def forward(self, fea1, fea2):     # implement as fid
        # mean
        mu1 = torch.mean(fea1, dim=0)
        mu2 = torch.mean(fea2, dim=0)
        m = torch.square(mu1 - mu2)
        # cov
        sigma1 = self.torch_cov(fea1)
        sigma2 = self.torch_cov(fea2)
        s = (sigma1 * sigma2).sqrt() * 2
        dis = m + (sigma1 + sigma2 - s).trace()
        return dis

