import torch
import torch.nn as nn
from einops import rearrange, repeat
from .contrast_and_atrous import AttnContrastLayer, AttnContrastLayer_n, \
    AtrousAttnWeight, AttnContrastLayer_d
import math
import torch.nn.functional as F

import numpy as np

__all__ = ['RPCANet9','RPCANet_LSTM']


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x

class RPCANet9(nn.Module):
    def __init__(self, stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32, mode='train'):
        super(RPCANet9, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(DecompositionModule9(slayers=slayers, llayers=llayers,
                                                  mlayers=mlayers, channel=channel))
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D, T = self.decos[i](D, T)
        if self.mode == 'train':
            return D,T
        else:
            return T

class DecompositionModule9(nn.Module):
    def __init__(self, slayers=6, llayers=3, mlayers=3, channel=32):
        super(DecompositionModule9, self).__init__()
        self.lowrank = LowrankModule9(channel=channel, layers=llayers)
        self.sparse = SparseModule9(channel=channel, layers=slayers)
        self.merge = MergeModule9(channel=channel, layers=mlayers)

    def forward(self, D, T):
        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T

class LowrankModule9(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule9, self).__init__()

        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)

    def forward(self, D, T):
        x = D - T
        B = x + self.convs(x)
        return B

class SparseModule9(nn.Module):
    def __init__(self, channel=32, layers=6) -> object:
        super(SparseModule9, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, B, T):
        x = T + D - B
        T = x - self.epsilon * self.convs(x)
        return T

class MergeModule9(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(MergeModule9, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D


class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel):
        super().__init__()
        pad_x = 1
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)

        pad_h = 1
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

    def forward(self, x, h, c):

        if h is None and c is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        return h, h, c

class RPCANet_LSTM(nn.Module):
    def __init__(self, stage_num=6, slayers=6, mlayers=3, channel=32, mode='train'):
        super(RPCANet_LSTM, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for i in range(stage_num):
            self.decos.append(DecompositionModule_LSTM(slayers=slayers, mlayers=mlayers, channel=channel))
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        [h, c] = [None, None]
        # B = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D, T, h, c = self.decos[i](D, T, h, c)
        if self.mode == 'train':
            return D,T
        else:
            return T

class DecompositionModule_LSTM(nn.Module):
    def __init__(self, slayers=6, mlayers=3, channel=32):
        super(DecompositionModule_LSTM, self).__init__()
        self.lowrank = LowrankModule_LSTM(channel=channel)
        self.sparse = SparseModule_LSTM(channel=channel, layers=slayers)
        self.merge = MergeModule_LSTM(channel=channel, layers=mlayers)

    def forward(self, D, T, h, c):
        B, h, c = self.lowrank(D, T, h, c)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T, h, c

class LowrankModule_LSTM(nn.Module):
    def __init__(self, channel=32):
        super(LowrankModule_LSTM, self).__init__()
        self.conv1_C = nn.Sequential(nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                         nn.BatchNorm2d(channel),
                         nn.ReLU(True))
        self.RB_1 = ResidualBlock(channel, channel, 3, bias=True, res_scale=1)
        self.RB_2 = ResidualBlock(channel, channel, 3, bias=True, res_scale=1)
        self.convC_1 = nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1)
        self.ConvLSTM = ConvLSTM(channel, channel, 3)

    def forward(self, D, T, h, c):
        x = D - T
        x_c = self.conv1_C(x)
        x_c1 = self.RB_1(x_c)
        x_ct, h, c = self.ConvLSTM(x_c1, h, c)
        x_c2 = self.RB_2(x_ct)
        x_1 = self.convC_1(x_c2)
        B = x + x_1
        return B, h, c

class SparseModule_LSTM(nn.Module):
    def __init__(self, channel=32, layers=6) -> object:
        super(SparseModule_LSTM, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)

        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.contrast = nn.Sequential(
                nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True),
                AttnContrastLayer_n(channel, kernel_size=17, padding=8),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1)
            )


    def forward(self, D, B, T):
        x = T + D - B
        w = self.contrast(x)
        T = x - self.epsilon * self.convs(x + w)
        return T

class MergeModule_LSTM(nn.Module):
    def __init__(self,  channel=32, layers=3):
        super(MergeModule_LSTM, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D


