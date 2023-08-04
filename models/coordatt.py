"""
Coordinate Attention for Efficient Mobile Network Design
https://github.com/houqb/CoordAttention/blob/main/coordatt.py
"""
import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''

        defaults = x.size()
        output_size = np.array([v if v is not None else d for v, d in zip(self.output_size, defaults[-len(self.output_size) :])])
        stride_size = np.floor(x.cpu().detach().numpy().shape[-2:] / output_size).astype(np.int32)
        kernel_size = x.cpu().detach().numpy().shape[-2:] - (output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x

class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=32):
        super(CoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h = AdaptiveAvgPool2dCustom((None, 1))
        self.pool_w = AdaptiveAvgPool2dCustom((1, None))
        
        mip = max(8, in_channels // reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identify = x
        N, C, H, W = x.size()
        x_h = self.pool_h(x)

        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # N,C,W,H

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identify * a_w * a_h

        return out
