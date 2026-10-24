"""ベースライン（linear スケジュール）を学習して、README の評価方法で測る。"""

import argparse

import numpy as np

from data import make_swiss_roll
from ddpm import get_betas, sample, train
from metrics import sliced_wasserstein

SEEDS = [0, 1, 2]


def main(steps: int):
    data = make_swiss_roll(2000, seed=0)  # 学習用
    ref = make_swiss_roll(2000, seed=100)  # 評価用の本物の点
    betas = get_betas("linear")
    scores = []
    for seed in SEEDS:
        model = train(betas, data, steps=steps, seed=seed)
        samples = sample(model, betas, n=2000, seed=seed)
        score = sliced_wasserstein(samples, ref)
        scores.append(score)
        print(f"seed {seed}: SW = {score:.3f}")
    print(f"linear: SW = {np.mean(scores):.3f} ± {np.std(scores):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000, help="学習のステップ数")
    main(parser.parse_args().steps)
