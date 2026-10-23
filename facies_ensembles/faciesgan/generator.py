import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EqualizedConv2d, NormalizationLayer, Upscale2dLayer

class Generator(nn.Module):
    def __init__(
        self,
        latent_shape,
        n_classes,
        *,
        normalize_latent=False,
        init_bias_to_zero=True,
        leaky_relu=0.2,
        equalized=True,
        per_channel_norm=True,
    ):
        super().__init__()
        self.latent_shape = list(latent_shape)
        self.n_classes = int(n_classes)
        self.normalize_latent = bool(normalize_latent)
        self.init_bias_to_zero = bool(init_bias_to_zero)
        self.equalized = bool(equalized)
        self.scales_depth = [self.latent_shape[0]]
        self.scale_layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.leaky = nn.LeakyReLU(leaky_relu)
        self.norm = NormalizationLayer() if per_channel_norm else None
        self.upscale = Upscale2dLayer()

    def add_scale(self, depth, kernel=3, n_extra=1):
        depth = int(depth)
        n_extra = max(0, int(n_extra))
        padding = kernel // 2
        in_ch = self.scales_depth[-1]
        self.scales_depth.append(depth)
        group = nn.ModuleList()
        group.append(
            EqualizedConv2d(
                in_ch,
                depth,
                kernelSize=kernel,
                padding=padding,
                equalized=self.equalized,
                initBiasToZero=self.init_bias_to_zero,
            )
        )

        for _ in range(n_extra):
            group.append(
                EqualizedConv2d(
                    depth,
                    depth,
                    kernelSize=kernel,
                    padding=padding,
                    equalized=self.equalized,
                    initBiasToZero=self.init_bias_to_zero,
                )
            )

        self.scale_layers.append(group)
        self.to_rgb.append(
            EqualizedConv2d(
                depth,
                self.n_classes,
                kernelSize=1,
                equalized=self.equalized,
                initBiasToZero=self.init_bias_to_zero,
            )
        )

    def output_size(self, scale=None):
        if scale is None:
            scale = len(self.to_rgb) - 1

        h = self.latent_shape[-2] * (2 ** scale)
        w = self.latent_shape[-1] * (2 ** scale)

        return h, w

    def _norm(self, x):
        if self.norm is None:
            return x

        return self.norm(x)

    def forward(self, z, scale=None, alpha=0.0):
        if scale is None:
            scale = len(self.to_rgb) - 1

        x = z

        if self.normalize_latent:
            x = self._norm(x)

        y = None

        for index, group in enumerate(self.scale_layers):
            if index > 0:
                x = self.upscale(x)

            if index <= scale:
                for conv in group:
                    x = self.leaky(conv(x))
                    x = self._norm(x)

                if alpha > 0 and index == (scale - 1):
                    y = self.upscale(self.to_rgb[index](x))

            if index == scale:
                x = self.to_rgb[index](x)

                if alpha > 0 and scale > 0 and y is not None:
                    x = alpha * y + (1.0 - alpha) * x

                return F.softmax(x, dim=1)

        return x
