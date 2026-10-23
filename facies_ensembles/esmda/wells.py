import numpy as np

def sample_well_ij(n_wells, height, width, rng=None):
    rng = np.random.default_rng(rng)
    chosen = set()
    out = []

    while len(out) < int(n_wells):
        i = int(rng.integers(0, height))
        j = int(rng.integers(0, width))

        if (i, j) in chosen:
            continue

        chosen.add((i, j))
        out.append((i, j))

    return out

def extract_well_labels(volume, wells):
    n, depth = volume.shape[0], volume.shape[1]
    obs = np.empty((len(wells) * depth, n), dtype=np.float64)
    row = 0

    for i, j in wells:
        for d in range(depth):
            obs[row, :] = volume[:, d, i, j]
            row += 1

    return obs

def reference_observations(reference_volume, wells):
    depth = reference_volume.shape[0]
    z = np.empty((len(wells) * depth, 1), dtype=np.float64)
    row = 0

    for i, j in wells:
        for d in range(depth):
            z[row, 0] = reference_volume[d, i, j]
            row += 1

    return z
