from pathlib import Path

import torch

from .build import build_discriminator, build_generator

def save_checkpoint(path, *, generator, discriminator=None, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": generator.state_dict(),
        "meta": dict(meta or {}),
    }

    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()

    torch.save(payload, path)

def load_generator(path, device="cpu", **build_kwargs):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "generator" in payload:
        meta = payload.get("meta") or {}
        kwargs = dict(build_kwargs)

        for key in ("n_classes", "latent_shape", "depths", "extras", "kernels"):
            if key in meta and key not in kwargs:
                kwargs[key] = meta[key]

        g = build_generator(**kwargs)
        g.load_state_dict(payload["generator"])

        return g.to(device), meta

    if isinstance(payload, dict) and "netG" in payload:
        meta = payload.get("meta") or {}
        kwargs = dict(build_kwargs)
        g = build_generator(**kwargs)
        g.load_state_dict(payload["netG"], strict=False)

        return g.to(device), meta

    raise ValueError(f"Unrecognized checkpoint format: {path}")

def load_discriminator(path, device="cpu", **build_kwargs):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(payload, dict) or "discriminator" not in payload:
        raise ValueError(f"No discriminator weights in {path}")

    meta = payload.get("meta") or {}
    kwargs = dict(build_kwargs)

    for key in ("n_classes", "latent_shape", "depths", "extras", "kernels", "fc_depth"):
        if key in meta and key not in kwargs:
            kwargs[key] = meta[key]

    d = build_discriminator(**kwargs)
    d.load_state_dict(payload["discriminator"])

    return d.to(device), meta
