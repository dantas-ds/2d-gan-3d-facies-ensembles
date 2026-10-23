import numpy as np

def facies_proportions(labels, n_classes=3):
    flat = np.asarray(labels).reshape(-1)

    return np.array(
        [(flat == k).mean() for k in range(int(n_classes))],
        dtype=np.float64,
    )
