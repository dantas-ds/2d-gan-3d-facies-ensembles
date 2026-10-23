#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facies_ensembles.faciesgan.checkpoint import load_generator
from facies_ensembles.latent.decode import decode_volume
from facies_ensembles.latent.trajectories import sample_trajectories

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rule", default="independent")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-control", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    g, meta = load_generator(args.checkpoint, device=args.device)
    latent_shape = tuple(meta.get("latent_shape", [4, 4, 4]))
    kwargs = {}

    if args.rule == "additive":
        kwargs["sigma"] = args.sigma

    if args.rule == "ldf":
        kwargs["n_control"] = args.n_control

    z = sample_trajectories(
        args.rule,
        args.n,
        args.depth,
        latent_shape,
        device=args.device,
        **kwargs,
    )
    scale = meta.get("scale")
    labels = decode_volume(g, z, scale=scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, labels.numpy().astype(np.int16))
    print(f"wrote {out} shape={tuple(labels.shape)}")

if __name__ == "__main__":
    main()
