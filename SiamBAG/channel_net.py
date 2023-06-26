import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Channel_attention_net(nn.Module):

    def __init__(self, channel=16, reduction=4):
        super(Channel_attention_net, self).__init__()
        # OrderedDict可以用于指定每个module的名字
        self.encoder = nn.Sequential(OrderedDict([
            ('encoder1', nn.Sequential(nn.Linear(channel, channel // 2, bias=True),
                                       nn.ReLU(inplace=False))),
            ('encoder2', nn.Sequential(nn.Linear(channel // 2, channel // 4, bias=True)))]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder1', nn.Sequential(nn.Linear(channel // 4, channel // 2, bias=True),
                                       nn.ReLU(inplace=False))),
            ('decoder2', nn.Sequential(nn.Linear(channel // 2, channel, bias=True)))]))
        self.soft = nn.ModuleList([nn.Sequential(nn.Softmax(dim=-1))])

    def forward(self, x):  # return 16 bands point-mul result
        b, c, w, h = x.size()
        # c1 = x.view(b, c, -1)
        c1 = x.reshape(b, c, -1)
        c2 = c1.permute(0, 2, 1)    # c2's size is [b,w*h,c]
        for name, module in self.encoder.named_children():
            c2 = module(c2)
        for name, module in self.decoder.named_children():
            c2 = module(c2)
        res2 = c2
        res2 = res2 / res2.max()
        res2 = self.soft[0](res2)
        res = res2.permute(0, 2, 1)
        # att = res.view(b, c, w, h)
        y = res.mean(dim=2)
        orderY = torch.sort(y, dim=-1, descending=True, out=None)
        # y = orderY[0]
        # y = y.view(b, c, 1, 1)
        # att = y.expand_as(x)
        # res = x.mul(att)
        # res = x
        return res, orderY
