# DDPM のノイズスケジュールの比較（2次元トイデータ）

拡散モデル（DDPM）で、ノイズスケジュールを **linear から cosine に変えると、生成の質が上がるか** を、
2次元のトイデータ（swiss roll）で調べる。

- 提案側：cosine スケジュール（Nichol & Dhariwal, 2021）
- ベースライン：linear スケジュール（Ho et al., 2020。β を 1e-4 から 0.02 まで線形に増やす）
- そろえる条件：データ、モデル（`ddpm.py` の `Denoiser`）、拡散のステップ数 `T`、学習のステップ数・学習率・バッチサイズ、サンプリングの方法は、両者で同じにする。変えるのはスケジュールだけ
- 評価：生成した 2,000 点と、本物の 2,000 点の **sliced Wasserstein 距離**（`metrics.py`、小さいほど良い）。データは元のスケールのまま測る。seed を 3 つ変えて、平均と標準偏差で比べる

## ファイル

- `data.py`：swiss roll の 2 次元データを作る
- `ddpm.py`：ノイズを予測する小さな MLP（`Denoiser`）、学習（`train`）、サンプリング（`sample`）。今は linear スケジュールだけ
- `metrics.py`：sliced Wasserstein 距離
