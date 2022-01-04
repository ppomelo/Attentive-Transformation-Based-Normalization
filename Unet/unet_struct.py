
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models
import torchvision
from functools import partial
import math

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, init_func, norm_func1, norm_func2, default1=False, default2=True):
        super(DoubleConv, self).__init__()
        self.conv = init_func(in_ch, out_ch, norm_func1, norm_func2, default1, default2)

    def forward(self, input):
        return self.conv(input)

class UpDoubConv(nn.Module):
    def __init__(self, in_ch, out_ch, init_func, norm_func1, norm_func2, default1=False, default2=False):
        super(UpDoubConv, self).__init__()
        self.conv = init_func(in_ch, out_ch, norm_func1, norm_func2, default1, default2)

    def forward(self, input):
        return self.conv(input)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Unet(nn.Module):
    def __init__(self, init_func, norm_func1, norm_func2):
        super(Unet, self).__init__()
        in_ch = 1
        out_ch = 1
        self.conv1 = UpDoubConv(in_ch, 32, init_func, norm_func1, norm_func2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64, init_func, norm_func1, norm_func2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128, init_func, norm_func1, norm_func2)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256, init_func, norm_func1, norm_func2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512, init_func, norm_func1, norm_func2)
        self.conv5 = UpDoubConv(256, 512, init_func, norm_func1, norm_func2)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256, init_func, norm_func1, norm_func2)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128, init_func, norm_func1, norm_func2)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64, init_func, norm_func1, norm_func2)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32, init_func, norm_func1, norm_func2)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x, target=False):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)

        # c5 = F.dropout2d(c5)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        out = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        
        return out