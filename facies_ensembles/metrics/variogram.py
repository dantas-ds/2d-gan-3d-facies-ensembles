import numpy as np

def indicator_variogram_horizontal(labels, lags, n_classes=3):
    labels = np.asarray(labels)

    if labels.ndim == 2:
        labels = labels[None, ...]

    if labels.ndim != 3:
        raise ValueError(f"Expected (N, H, W) or (H, W), got {labels.shape}")

    lags = np.asarray(lags, dtype=int)
    out = np.zeros((int(n_classes), len(lags)), dtype=np.float64)

    for k in range(int(n_classes)):
        ind = (labels == k).astype(np.float32)

        for j, h in enumerate(lags):
            h = int(h)
            dx = np.mean((ind[:, :, h:] - ind[:, :, :-h]) ** 2)
            dy = np.mean((ind[:, h:, :] - ind[:, :-h, :]) ** 2)
            out[k, j] = 0.25 * (dx + dy)

    return out

def vertical_indicator_variogram(volumes, lags, n_classes=3):
    volumes = np.asarray(volumes)

    if volumes.ndim == 3:
        volumes = volumes[None, ...]

    if volumes.ndim != 4:
        raise ValueError(f"Expected (N, D, H, W) or (D, H, W), got {volumes.shape}")

    lags = np.asarray(lags, dtype=int)
    depth = volumes.shape[1]
    out = np.zeros((int(n_classes), len(lags)), dtype=np.float64)

    for k in range(int(n_classes)):
        ind = (volumes == k).astype(np.float32)

        for j, h in enumerate(lags):
            h = int(h)

            if h >= depth:
                out[k, j] = np.nan
                continue

            diff = (ind[:, h:] - ind[:, :-h]) ** 2
            out[k, j] = 0.5 * float(diff.mean())

    return out

def vertical_variogram_rmse(gen_vgram, ref_vgram):
    a = np.asarray(gen_vgram, dtype=np.float64)
    b = np.asarray(ref_vgram, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)

    if not np.any(m):
        return np.nan

    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
