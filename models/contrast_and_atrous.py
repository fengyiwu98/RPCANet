import torch
import torch.nn as nn
import torch.nn.functional as F


class Avg_ChannelAttention_n(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention_n, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 bz,C_out,h,w -> bz,C_out,1,1
            nn.Conv2d(channels, channels // r, 1, 1, 0),  # bz,C_out,1,1 -> bz,C_out/r,1,1
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.avg_channel(x)



class Avg_ChannelAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 bz,C_out,h,w -> bz,C_out,1,1
            nn.Conv2d(channels, channels // r, 1, 1, 0),  # bz,C_out,1,1 -> bz,C_out/r,1,1
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.avg_channel(x)


class AttnContrastLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(AttnContrastLayer, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.attn = Avg_ChannelAttention(channels)

    def forward(self, x):

        out_normal = self.conv(x)

        theta = self.attn(x)

        kernel_w1 = self.conv.weight.sum(2).sum(2)

        kernel_w2 = kernel_w1[:, :, None, None]

        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        return theta * out_center - out_normal



class AttnContrastLayer_n(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(AttnContrastLayer_n, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.attn = Avg_ChannelAttention_n(channels)

    def forward(self, x):
        out_normal = self.conv(x)
        theta = self.attn(x)


        kernel_w1 = self.conv.weight.sum(2).sum(2)
        kernel_w2 = kernel_w1[:, :, None, None]

        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        return theta * out_center - out_normal

class AttnContrastLayer_d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=2, groups=1, bias=False):
        super(AttnContrastLayer_d, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.attn = Avg_ChannelAttention(channels)

    def forward(self, x):

        out_normal = self.conv(x)

        theta = self.attn(x)

        kernel_w1 = self.conv.weight.sum(2).sum(2)
        kernel_w2 = kernel_w1[:, :, None, None]
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        return out_center - theta * out_normal

class AtrousAttnWeight(nn.Module):
    def __init__(self, channels):
        super(AtrousAttnWeight, self).__init__()
        self.attn = Avg_ChannelAttention(channels)

    def forward(self, x):
        return self.attn(x)


