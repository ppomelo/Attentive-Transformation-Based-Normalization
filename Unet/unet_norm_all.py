
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models
import torchvision
from functools import partial
import math

class AttentiveTrans2d(nn.Module):

    def __init__(self, num_features, hidden_channels=32):
        super(AttentiveTrans2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.smooth_gamma = 1
        self.smooth_beta = 0
        self.matrix1 = nn.Parameter(torch.ones(num_features, hidden_channels))
        self.matrix2 = nn.Parameter(torch.ones(hidden_channels, num_features))
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(num_features, num_features, 1, bias=False)
        self.conv4 = nn.Conv2d(num_features, num_features, 1, bias=False)
        self.IN_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)


    def forward(self, feature):
 
        output = self.IN_norm(feature)

        feature_nc = self.avgpool(feature).view(feature.size()[0], feature.size()[1])
        channel_wise_response = self.sigmoid(feature_nc@self.matrix1)@self.matrix2
        channel_wise_response = channel_wise_response.unsqueeze(-1).unsqueeze(-1).expand(output.size())

        avg_out = F.adaptive_avg_pool3d(feature,(1,feature.size()[2],feature.size()[3]))
        max_out = F.adaptive_max_pool3d(feature,(1,feature.size()[2],feature.size()[3]))
        avg_max_concat = torch.cat([avg_out, max_out], dim=1)
        
        spatial_wise_response = self.conv2(self.sigmoid(self.conv1(avg_max_concat))).expand(output.size())

        pixel_wise_response = channel_wise_response * spatial_wise_response

        importance_gamma = self.conv3(pixel_wise_response) + self.smooth_gamma
        importance_beta = self.conv4(pixel_wise_response) + self.smooth_beta
        
        out_in = output * importance_gamma + importance_beta

        return out_in

class InstanceEnhancementBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(InstanceEnhancementBatchNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias_readjust = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input):
        self._check_input_dim(input)

        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * attention

        out_bn = F.batch_norm(input, self.running_mean, self.running_var, None, None, self.training, self.momentum,
                              self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn

class LocalContextNorm(nn.Module):
    def __init__(self, num_features, channels_per_group=2, window_size=(227, 227), eps=1e-5):
        super(LocalContextNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size

    def forward(self, x):
        N, C, H, W = x.size()
        G = C // self.channels_per_group
        assert C % self.channels_per_group == 0
        if self.window_size[0] < H and self.window_size[1] < W:
            # Build integral image
            device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
            x_squared = x * x
            integral_img = x.cumsum(dim=2).cumsum(dim=3)
            integral_img_sq = x_squared.cumsum(dim=2).cumsum(dim=3)
            # Dilation
            d = (1, self.window_size[0], self.window_size[1])
            integral_img = torch.unsqueeze(integral_img, dim=1)
            integral_img_sq = torch.unsqueeze(integral_img_sq, dim=1)
            kernel = torch.tensor([[[[[1., -1.], [-1., 1.]]]]]).to(device)
            c_kernel = torch.ones((1, 1, self.channels_per_group, 1, 1)).to(device)
            with torch.no_grad():
                # Dilated conv
                sums = F.conv3d(integral_img, kernel, stride=[1, 1, 1], dilation=d)
                sums = F.conv3d(sums, c_kernel, stride=[self.channels_per_group, 1, 1])
                squares = F.conv3d(integral_img_sq, kernel, stride=[1, 1, 1], dilation=d)
                squares = F.conv3d(squares, c_kernel, stride=[self.channels_per_group, 1, 1])
            n = self.window_size[0] * self.window_size[1] * self.channels_per_group
            means = torch.squeeze(sums / n, dim=1)
            var = torch.squeeze((1.0 / n * (squares - sums * sums / n)), dim=1)
            _, _, h, w = means.size()
            pad2d = (int(math.floor((W - w) / 2)), int(math.ceil((W - w) / 2)), int(math.floor((H - h) / 2)),
                     int(math.ceil((H - h) / 2)))
            padded_means = F.pad(means, pad2d, 'replicate')
            padded_vars = F.pad(var, pad2d, 'replicate') + self.eps
            for i in range(G):
                x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] = \
                    (x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] -
                     torch.unsqueeze(padded_means[:, i, :, :], dim=1).to(device)) /\
                    ((torch.unsqueeze(padded_vars[:, i, :, :], dim=1)).to(device)).sqrt()
            del integral_img
            del integral_img_sq
        else:
            x = x.view(N, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)

        return x * self.weight + self.bias

class AttentiveNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, hidden_channels=32, eps=1e-5, momentum=0.1, track_running_stats=False):
        super(AttentiveNorm2d, self).__init__(num_features,
                                              eps=eps,
                                              momentum=momentum,
                                              affine=False,
                                              track_running_stats=track_running_stats)

        self.gamma = nn.Parameter(torch.randn(hidden_channels, num_features))
        self.beta = nn.Parameter(torch.randn(hidden_channels, num_features))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, hidden_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = super(AttentiveNorm2d, self).forward(x)

        size = output.size()
        b, c, _, _ = x.size()

        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)

        gamma = y @ self.gamma
        beta = y @ self.beta

        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)

        return gamma * output + beta
