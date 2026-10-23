# Checkpoints

Generator weights used for the manuscript figures and experiments.

| File | Dataset |
|------|---------|
| [`ganriver_i_s5_i460000.pt`](ganriver_i_s5_i460000.pt) | GANRiver-I |
| [`petrel_s5_i580000.pt`](petrel_s5_i580000.pt) | Petrel |

```bash
python scripts/build_volumes.py \
  --checkpoint checkpoints/ganriver_i_s5_i460000.pt \
  --rule ldf --n 4 --depth 32 \
  --out outputs/volumes.npy
```
