import numpy as np

def connectivity_3d(volume, facies=2):
    from scipy import ndimage

    volume = np.asarray(volume)

    if volume.ndim != 3:
        raise ValueError(f"Expected (D, H, W), got {volume.shape}")

    mask = volume == int(facies)
    n_cells = int(mask.sum())

    if n_cells == 0:
        return {
            "n_channel": 0,
            "n_largest": 0,
            "c3d": np.nan,
            "n_components": 0,
            "channel_prop": 0.0,
        }

    structure = ndimage.generate_binary_structure(3, 1)
    labeled, ncomp = ndimage.label(mask, structure=structure)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, ncomp + 1))
    n_largest = int(np.max(sizes)) if ncomp > 0 else 0

    return {
        "n_channel": n_cells,
        "n_largest": n_largest,
        "c3d": float(n_largest / n_cells),
        "n_components": int(ncomp),
        "channel_prop": float(n_cells / mask.size),
    }
