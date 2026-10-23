from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SCALE_KEY = "scale_{}"

def write_scales_h5(path, arrays, *, n_classes=None, overwrite=False):
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        if n_classes is not None:
            f.attrs["n_classes"] = int(n_classes)

        f.attrs["n_scales"] = int(len(arrays))

        for i, arr in enumerate(arrays):
            a = np.asarray(arr)

            if a.ndim != 3:
                raise ValueError(f"scale {i} must be (N,H,W), got {a.shape}")

            f.create_dataset(SCALE_KEY.format(i), data=a, compression="gzip")

def open_scale(path, scale):
    path = Path(path)
    key = SCALE_KEY.format(int(scale))
    f = h5py.File(path, "r")

    if key not in f:
        f.close()
        raise KeyError(f"{path} missing dataset {key}")

    return f, f[key]

class ScaleH5Dataset(Dataset):
    def __init__(self, path, scale, *, n_classes, one_hot=True):
        self.path = Path(path)
        self.scale = int(scale)
        self.n_classes = int(n_classes)
        self.one_hot = bool(one_hot)

        with h5py.File(self.path, "r") as f:
            ds = f[SCALE_KEY.format(self.scale)]
            self.n = int(ds.shape[0])
            self.h = int(ds.shape[1])
            self.w = int(ds.shape[2])

        self._file = None
        self._ds = None

    def _ensure(self):
        if self._ds is None:
            self._file = h5py.File(self.path, "r")
            self._ds = self._file[SCALE_KEY.format(self.scale)]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self._ensure()
        labels = np.asarray(self._ds[index], dtype=np.int64)

        if not self.one_hot:
            return torch.from_numpy(labels)

        out = np.zeros((self.n_classes, self.h, self.w), dtype=np.float32)

        for c in range(self.n_classes):
            out[c] = labels == c

        return torch.from_numpy(out)

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
