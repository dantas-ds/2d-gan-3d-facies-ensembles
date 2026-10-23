import torch

def gradient_penalty(discriminator, real, fake, *, scale, alpha, weight=10.0):
    batch = real.size(0)
    eps = torch.rand(batch, 1, 1, 1, device=real.device, dtype=real.dtype)
    inter = eps * real + (1.0 - eps) * fake
    inter.requires_grad_(True)
    score = discriminator(inter, scale=scale, alpha=alpha)

    grads = torch.autograd.grad(
        outputs=score,
        inputs=inter,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.reshape(batch, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    return weight * gp

def discriminator_loss(real_score, fake_score, *, drift=0.001):
    return (
        fake_score.mean()
        - real_score.mean()
        + drift * (real_score ** 2).mean()
    )

def generator_loss(fake_score):
    return -fake_score.mean()
