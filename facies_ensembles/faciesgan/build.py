from .discriminator import Discriminator
from .generator import Generator

DEFAULT_DEPTHS = [128, 128, 128, 128, 64, 32]
DEFAULT_EXTRAS = [0, 1, 1, 1, 1, 2]
DEFAULT_KERNELS = [3, 3, 3, 3, 3, 3]
DEFAULT_LATENT = [4, 4, 4]

def build_generator(
    n_classes=3,
    *,
    latent_shape=None,
    depths=None,
    extras=None,
    kernels=None,
    **kwargs,
):
    latent_shape = list(latent_shape or DEFAULT_LATENT)
    depths = list(depths or DEFAULT_DEPTHS)
    extras = list(extras or DEFAULT_EXTRAS)
    kernels = list(kernels or DEFAULT_KERNELS)

    g = Generator(latent_shape, n_classes, **kwargs)

    for i, depth in enumerate(depths):
        g.add_scale(depth, kernel=kernels[i], n_extra=extras[i])

    return g

def build_discriminator(
    n_classes=3,
    *,
    latent_shape=None,
    depths=None,
    extras=None,
    kernels=None,
    fc_depth=256,
    **kwargs,
):
    latent_shape = list(latent_shape or DEFAULT_LATENT)
    depths = list(depths or DEFAULT_DEPTHS)
    extras = list(extras or DEFAULT_EXTRAS)
    kernels = list(kernels or DEFAULT_KERNELS)

    d = Discriminator(
        latent_shape,
        n_classes,
        first_depth=depths[0],
        fc_depth=fc_depth,
        **kwargs,
    )

    for i, depth in enumerate(depths):
        d.add_scale(depth, kernel=kernels[i], n_extra=extras[i])

    return d
