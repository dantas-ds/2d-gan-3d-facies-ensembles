#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facies_ensembles.train.trainer import ProgressiveTrainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--start-scale", type=int, default=0)
    p.add_argument("--end-scale", type=int, default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer = ProgressiveTrainer(cfg)
    trainer.train(start_scale=args.start_scale, end_scale=args.end_scale)

if __name__ == "__main__":
    main()
