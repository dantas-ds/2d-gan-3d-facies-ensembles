#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facies_ensembles.esmda.run import run_esmda
from facies_ensembles.faciesgan.checkpoint import load_generator

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--rule", default="ldf")
    p.add_argument("--n-ens", type=int, default=32)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--n-wells", type=int, default=4)
    p.add_argument("--n-assimilations", type=int, default=4)
    p.add_argument("--obs-variance", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-control", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    g, meta = load_generator(args.checkpoint, device=args.device)
    latent_shape = tuple(meta.get("latent_shape", [4, 4, 4]))
    ref = np.load(args.reference)

    if ref.ndim == 4:
        ref = ref[0]

    rule_kwargs = {}

    if args.rule == "additive":
        rule_kwargs["sigma"] = args.sigma

    if args.rule == "ldf":
        rule_kwargs["n_control"] = args.n_control

    result = run_esmda(
        g,
        rule=args.rule,
        n_ens=args.n_ens,
        depth=args.depth,
        latent_shape=latent_shape,
        reference_volume=ref,
        n_wells=args.n_wells,
        n_assimilations=args.n_assimilations,
        obs_variance=args.obs_variance,
        seed=args.seed,
        device=args.device,
        rule_kwargs=rule_kwargs,
        scale=meta.get("scale"),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "labels_post.npy", result["labels_post"].numpy().astype(np.int16))
    torch.save(result["z_post"], out / "z_post.pt")

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump({"wells": result["wells"], "history": result["history"]}, f, indent=2)

    print(f"wrote {out}")

if __name__ == "__main__":
    main()
