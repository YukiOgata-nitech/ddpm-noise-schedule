"""生成した点群と本物の点群の距離。"""

import numpy as np


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_projections: int = 128, seed: int = 0) -> float:
    """2 つの点群（同じ点数）の sliced Wasserstein 距離。小さいほど分布が近い。"""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_projections, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)
    return float(np.mean(np.abs(pa - pb)))
