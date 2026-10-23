import math

import torch
import torch.nn as nn
from numpy import prod

class NormalizationLayer(nn.Module):
    def forward(self, x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

class ConstrainedLayer(nn.Module):
    def __init__(self, module, equalized=True, gain=2.0, initBiasToZero=True):
        super().__init__()
        self.module = module
        self.equalized = equalized

        if self.module is not None:
            if initBiasToZero:
                self.module.bias.data.fill_(0)

            if equalized:
                self.module.weight.data.normal_(0, 1)
                size = self.module.weight.size()
                fan_in = prod(size[1:])
                self.weight = math.sqrt(gain / fan_in)

    def forward(self, x):
        if self.module is None:
            return x

        if self.equalized:
            return self.module(x * self.weight)

        return self.module(x)

class EqualizedConv2d(ConstrainedLayer):
    def __init__(
        self,
        nChannelsPrevious,
        nChannels,
        kernelSize,
        padding=0,
        bias=True,
        **kwargs,
    ):
        module = None

        if nChannels > 0:
            module = nn.Conv2d(
                nChannelsPrevious,
                nChannels,
                kernelSize,
                padding=padding,
                bias=bias,
            )

        super().__init__(module, **kwargs)

class EqualizedLinear(ConstrainedLayer):
    def __init__(self, nChannelsPrevious, nChannels, bias=True, **kwargs):
        super().__init__(
            nn.Linear(nChannelsPrevious, nChannels, bias=bias),
            **kwargs,
        )

class Upscale2dLayer(nn.Module):
    def forward(self, x, factor=2):
        if factor == 1:
            return x

        s = x.size()
        x = x.view(-1, s[1], s[2], 1, s[3], 1)
        x = x.expand(-1, s[1], s[2], factor, s[3], factor)

        return x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)

class MiniBatchStdDev(nn.Module):
    def forward(self, x, subGroupSize=4):
        size = x.size()
        subGroupSize = min(size[0], subGroupSize)

        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]

        G = int(size[0] / subGroupSize)

        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        return torch.cat([x, y], dim=1)
