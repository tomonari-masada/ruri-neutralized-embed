#!/usr/bin/env python3
"""
ruri-v3-310m_gender_pairs.py の結果（PC1 の寄与率など）を検証するための追加分析。
ruri-v3-310m_gender_pairs.json / ruri-v3-310m_gender_pairs.py /
ruri-v3-310m_gender_pairs_diff.npz と同じフォルダで実行する。

  python ruri-v3-310m_gender_pairs_checks.py
  python ruri-v3-310m_gender_pairs_checks.py --safetensors /path/to/model.safetensors --tokenizer-json /path/to/tokenizer.json
"""
import argparse
import importlib
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="ruri-v3-310m_gender_pairs_diff.npz")
ap.add_argument("--safetensors", default=None)
ap.add_argument("--tokenizer-json", default=None)
ap.add_argument("--n-random", type=int, default=50)
args = ap.parse_args()

m = importlib.import_module("ruri-v3-310m_gender_pairs")  # 同じフォルダの .py を読み込む
W, _ = m.load_unembedding(safetensors_path=args.safetensors)
r = np.load(args.npz)
D = r["diff_unit"]
N, d = D.shape
cat = r["category"]


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def pc1_ratio(X):
    s = np.linalg.svd(X, compute_uv=False)
    return s[0] ** 2 / (s ** 2).sum()


# 語彙のスコア（Unigram の対数確率 ≒ 対数頻度）
if args.tokenizer_json is None:
    from huggingface_hub import hf_hub_download
    args.tokenizer_json = hf_hub_download(m.MODEL_ID, "tokenizer.json")
vocab = json.load(open(args.tokenizer_json, encoding="utf-8"))["model"]["vocab"]
toks = [t for t, _ in vocab]
score = np.array([s for _, s in vocab])
normal = np.where(score < 0)[0]  # 特殊トークン（score=0）を除く

# (1) ランダムなトークン対での PC1 寄与率（帰無ベースライン）
rng = np.random.default_rng(0)
base = []
for _ in range(args.n_random):
    a, b = rng.choice(normal, (2, N))
    base.append(pc1_ratio(unit(W[a] - W[b])))
print(f"[1] PC1 寄与率: 男女ペア {pc1_ratio(D):.3f}  /  ランダム対 平均 {np.mean(base):.3f}, 最大 {np.max(base):.3f}")

# (1b) 中心化した PCA（平均方向を除いた後のばらつきの構造）
print(f"    中心化後の PC1 寄与率: {pc1_ratio(D - D.mean(0)):.3f}")

# (2) カテゴリを 1 つ抜いて方向を推定し、抜いたカテゴリで検証（変化形の重複による漏れを防ぐ）
print("[2] leave-one-category-out: 残りから推定した平均方向と、抜いたカテゴリの差ベクトルとの cos")
for c in np.unique(cat):
    g = unit(D[cat != c].mean(0))
    cs = D[cat == c] @ g
    print(f"    {c:<18} n={len(cs):>3}  mean cos={cs.mean():.3f}  min={cs.min():.3f}")

# (3) 頻度の交絡: 性別方向がトークン頻度の方向と重なっていないか
g = r["gender_dir_pc1"]  # 正 = 男性側
proj = unit(W) @ g
print(f"[3] 語彙全体での corr(性別方向への射影, 対数頻度) = {np.corrcoef(proj[normal], score[normal])[0, 1]:.3f}")
dscore = score[r["male_ids"]] - score[r["female_ids"]]
print(f"    ペアごとの corr(cos_to_mean, 男性側−女性側の対数頻度差) = {np.corrcoef(r['cos_to_mean'], dscore)[0, 1]:.3f}")
print(f"    男性側の方が高頻度のペアの割合 = {(dscore > 0).mean():.2f}")

# (4) 性別方向の両端にある語彙（定性的な確認）。方向の推定に使ったペアのトークンは除外する
used = set(r["male_ids"].tolist()) | set(r["female_ids"].tolist())
cand = np.array([i for i in normal if i not in used])
o = cand[np.argsort(proj[cand])]
print("[4] 女性側の端:", " ".join(toks[i] for i in o[:40]))
print("    男性側の端:", " ".join(toks[i] for i in o[::-1][:40]))
