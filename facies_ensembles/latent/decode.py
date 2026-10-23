import torch

@torch.no_grad()
def decode_volume(generator, z_chain, *, scale=None, batch_slices=32):
    n, depth = z_chain.shape[0], z_chain.shape[1]
    device = next(generator.parameters()).device

    generator.eval()

    slices = []
    flat = z_chain.reshape(n * depth, *z_chain.shape[2:]).to(
        device=device, dtype=torch.float32
    )

    for start in range(0, flat.shape[0], batch_slices):
        chunk = flat[start : start + batch_slices]
        prob = generator(chunk, scale=scale, alpha=0.0)
        labels = prob.argmax(dim=1)
        slices.append(labels.cpu())

    labels = torch.cat(slices, dim=0).reshape(n, depth, *slices[0].shape[1:])

    return labels
