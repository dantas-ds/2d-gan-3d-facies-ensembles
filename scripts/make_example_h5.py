#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facies_ensembles.data.h5_dataset import write_scales_h5

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/example_scales.h5")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--n-scales", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    arrays = []

    for s in range(args.n_scales):
        side = 4 * (2 ** s)
        arrays.append(
            rng.integers(0, args.n_classes, size=(args.n, side, side), dtype=np.int64)
        )

    write_scales_h5(args.out, arrays, n_classes=args.n_classes, overwrite=True)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
