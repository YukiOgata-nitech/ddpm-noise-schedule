"""swiss roll の 2 次元データ。"""

import numpy as np


def make_swiss_roll(n: int = 2000, noise: float = 0.5, seed: int = 0) -> np.ndarray:
    """(n, 2) の配列を返す。座標はおおよそ -10〜15 の範囲（元のスケール）。"""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    points = np.stack([x, y], axis=1)
    points += noise * rng.standard_normal(points.shape)
    return points.astype(np.float32)
