import torch
import torch.nn as nn
from numpy import prod

from .layers import EqualizedConv2d, EqualizedLinear, MiniBatchStdDev

class Discriminator(nn.Module):
    def __init__(
        self,
        latent_shape,
        n_classes,
        *,
        first_depth,
        fc_depth=256,
        init_bias_to_zero=True,
        leaky_relu=0.2,
        equalized=True,
        minibatch_std=True,
    ):
        super().__init__()
        self.latent_shape = list(latent_shape)
        self.n_classes = int(n_classes)
        self.init_bias_to_zero = bool(init_bias_to_zero)
        self.equalized = bool(equalized)
        self.minibatch_std = bool(minibatch_std)
        self.fc_depth = int(fc_depth)
        self.scales_depth = [int(first_depth)]
        self.scale_layers = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        self.last = nn.ModuleList(
            [
                EqualizedLinear(
                    int(first_depth) * prod(latent_shape[-2:]),
                    self.fc_depth,
                    equalized=equalized,
                    initBiasToZero=init_bias_to_zero,
                ),
                EqualizedLinear(
                    self.fc_depth,
                    1,
                    equalized=equalized,
                    initBiasToZero=init_bias_to_zero,
                ),
            ]
        )
        self.mbstd = MiniBatchStdDev()
        self.leaky = nn.LeakyReLU(leaky_relu)

    def add_scale(self, depth, kernel=3, n_extra=1):
        depth = int(depth)
        n_extra = max(0, int(n_extra))
        padding = kernel // 2
        depth_last = self.scales_depth[-1]
        self.scales_depth.append(depth)
        group = nn.ModuleList()
        entry = depth

        if self.minibatch_std and len(self.scale_layers) == 0:
            entry = depth + 1

        for i_extra in range(n_extra):
            group.append(
                EqualizedConv2d(
                    entry,
                    depth,
                    kernel,
                    padding=padding,
                    equalized=self.equalized,
                    initBiasToZero=self.init_bias_to_zero,
                )
            )

            if i_extra == 0 and self.minibatch_std and len(self.scale_layers) == 0:
                entry = depth

        group.append(
            EqualizedConv2d(
                entry,
                depth_last,
                kernel,
                padding=padding,
                equalized=self.equalized,
                initBiasToZero=self.init_bias_to_zero,
            )
        )
        self.scale_layers.append(group)
        self.from_rgb.append(
            EqualizedConv2d(
                self.n_classes,
                depth,
                1,
                equalized=self.equalized,
                initBiasToZero=self.init_bias_to_zero,
            )
        )

    def forward(self, x, scale=None, alpha=0.0):
        if scale is None:
            scale = len(self.from_rgb) - 1

        reverse = len(self.from_rgb) - 1 - scale

        for index, group in enumerate(reversed(self.scale_layers)):
            if index == reverse:
                z = x
                x = self.leaky(self.from_rgb[-index - 1](x))

                if alpha > 0 and scale > 0:
                    y = nn.AvgPool2d((2, 2))(z)
                    y = self.leaky(self.from_rgb[-index - 2](y))

            if index >= reverse:
                if self.minibatch_std and index == len(self.from_rgb) - 1:
                    x = self.mbstd(x)

                for layer in group:
                    x = self.leaky(layer(x))

                if index != len(self.from_rgb) - 1:
                    x = nn.AvgPool2d((2, 2))(x)

                if index == reverse and alpha > 0 and scale > 0:
                    x = alpha * y + (1.0 - alpha) * x

        x = x.view(-1, prod(x.size()[1:]))
        x = self.leaky(self.last[-2](x))

        return self.last[-1](x)
