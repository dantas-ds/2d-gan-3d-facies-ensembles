import torch
import torch.nn.functional as F

def sample_independent(n, depth, latent_shape, *, device="cpu", generator=None):
    shape = (int(n), int(depth), *tuple(latent_shape))

    if generator is None:
        return torch.randn(shape, device=device)

    return torch.randn(shape, device=device, generator=generator)

def _lerp(z0, z1, t):
    return (1.0 - t) * z0 + t * z1

def _slerp(z0, z1, t):
    a = z0.reshape(z0.shape[0], -1)
    b = z1.reshape(z1.shape[0], -1)
    a_n = a / (a.norm(dim=1, keepdim=True) + 1e-10)
    b_n = b / (b.norm(dim=1, keepdim=True) + 1e-10)
    dot = (a_n * b_n).sum(dim=1).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    out = torch.empty_like(z0)
    small = so.abs() < 1e-6

    if small.any():
        out[small] = _lerp(z0[small], z1[small], t)

    ok = ~small

    if ok.any():
        t = torch.as_tensor(t, device=z0.device, dtype=z0.dtype)
        c1 = torch.sin((1.0 - t) * omega[ok]) / so[ok]
        c2 = torch.sin(t * omega[ok]) / so[ok]
        out[ok] = (
            c1[:, None, None, None] * z0[ok] + c2[:, None, None, None] * z1[ok]
        )

    return out

def sample_lerp(n, depth, latent_shape, *, device="cpu", generator=None):
    z0 = sample_independent(n, 1, latent_shape, device=device, generator=generator)[:, 0]
    z1 = sample_independent(n, 1, latent_shape, device=device, generator=generator)[:, 0]
    ts = torch.linspace(0.0, 1.0, int(depth), device=device)
    out = torch.empty(int(n), int(depth), *latent_shape, device=device)

    for i, t in enumerate(ts):
        out[:, i] = _lerp(z0, z1, float(t))

    return out

def sample_slerp(n, depth, latent_shape, *, device="cpu", generator=None):
    z0 = sample_independent(n, 1, latent_shape, device=device, generator=generator)[:, 0]
    z1 = sample_independent(n, 1, latent_shape, device=device, generator=generator)[:, 0]
    ts = torch.linspace(0.0, 1.0, int(depth), device=device)
    out = torch.empty(int(n), int(depth), *latent_shape, device=device)

    for i, t in enumerate(ts):
        out[:, i] = _slerp(z0, z1, float(t))

    return out

def sample_additive(
    n,
    depth,
    latent_shape,
    *,
    sigma=0.1,
    device="cpu",
    generator=None,
):
    out = torch.empty(int(n), int(depth), *latent_shape, device=device)
    out[:, 0] = sample_independent(n, 1, latent_shape, device=device, generator=generator)[:, 0]

    for d in range(1, int(depth)):
        eps = sample_independent(n, 1, latent_shape, device=device, generator=generator)[:, 0]
        out[:, d] = out[:, d - 1] + float(sigma) * eps

    return out

def sample_ldf(
    n,
    depth,
    latent_shape,
    *,
    n_control=4,
    device="cpu",
    generator=None,
):
    d = int(depth)
    dc = max(1, min(int(n_control), d))
    coarse = sample_independent(n, dc, latent_shape, device=device, generator=generator)

    if dc == d:
        return coarse

    c, h, w = int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2])
    flat = coarse.permute(0, 2, 3, 4, 1).reshape(int(n), c * h * w, dc)
    flat = F.interpolate(flat, size=d, mode="linear", align_corners=True)

    return flat.reshape(int(n), c, h, w, d).permute(0, 4, 1, 2, 3).contiguous()

RULES = {
    "independent": sample_independent,
    "lerp": sample_lerp,
    "slerp": sample_slerp,
    "additive": sample_additive,
    "ldf": sample_ldf,
}

def sample_trajectories(rule, n, depth, latent_shape, **kwargs):
    key = str(rule).lower().strip()

    if key not in RULES:
        raise ValueError(f"Unknown rule {rule!r}. Choose from {sorted(RULES)}")

    return RULES[key](n, depth, latent_shape, **kwargs)
