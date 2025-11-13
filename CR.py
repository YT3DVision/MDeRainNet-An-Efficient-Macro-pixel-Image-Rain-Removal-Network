import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        #vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg_19 = models.vgg19(pretrained=False)
        vgg_19.load_state_dict(torch.load('./vgg/vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg_19.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        loss_total = []
        h = p.shape[2] // 5
        w = p.shape[3] // 5
        for row in range(5):
            ibx = row * h
            iex = ibx + h
            for col in range(5):
                iby = col * w
                iey = iby + w
                pred_im_features = a[:, :, ibx:iex, iby:iey]
                gt_features = p[:, :, ibx:iex, iby:iey]
                input_features = n[:, :, ibx:iex, iby:iey]

                a_vgg, p_vgg, n_vgg = self.vgg(pred_im_features), self.vgg(gt_features), self.vgg(input_features)
                loss = 0

                d_ap, d_an = 0, 0
                for i in range(len(a_vgg)):
                    d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
                    if not self.ab:
                        d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                        contrastive = d_ap / (d_an + 1e-7)
                    else:
                        contrastive = d_ap

                    loss += self.weights[i] * contrastive
                loss_total.append(loss)
        return sum(loss_total) / len(loss_total)
