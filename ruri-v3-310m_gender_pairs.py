#!/usr/bin/env python3
"""
cl-nagoya/ruri-v3-310m の unembedding 行列から、男女サブワードペアの
unembedding ベクトルを取り出し、その差 (male - female) を計算する。

入力 : ruri-v3-310m_gender_pairs.json（男女ペアの一覧）
出力 : ruri-v3-310m_gender_pairs_diff.npz（差ベクトルなど）

■ unembedding 行列について
  ruri-v3-310m の model.safetensors には MLM ヘッド（head.dense / head.norm /
  decoder.bias）が含まれず、重みは embeddings.tok_embeddings.weight
  (102400 x 768) のみ。ModernBERT は tie_word_embeddings=True（既定）で
  decoder.weight を tok_embeddings.weight と共有するため、この行列を
  unembedding 行列として用いる。
  （decoder.weight / lm_head.weight を別に持つチェックポイントを
    --safetensors で与えた場合はそちらを優先する。）
  decoder.bias はスカラーのトークン別バイアスなので、ベクトルの差には関与しない。

使い方:
  pip install numpy safetensors huggingface_hub   # 検証用に transformers も推奨
  python ruri-v3-310m_gender_pairs.py
  python ruri-v3-310m_gender_pairs.py --strict-only
  python ruri-v3-310m_gender_pairs.py --lang ja --strict-only --out ja_strict.npz
  python ruri-v3-310m_gender_pairs.py --safetensors /path/to/model.safetensors --no-verify
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_ID = "cl-nagoya/ruri-v3-310m"
PAIRS_JSON = "ruri-v3-310m_gender_pairs.json"

# 優先順：独立した出力層があればそれ、なければ（重み共有の）入力埋め込み
UNEMBED_KEYS = [
    "decoder.weight",
    "lm_head.weight",
    "embeddings.tok_embeddings.weight",
    "model.embeddings.tok_embeddings.weight",
]


def load_pairs(path: str, strict_only: bool = False, lang: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    if strict_only:
        pairs = [p for p in pairs if p["strict"]]
    if lang:
        pairs = [p for p in pairs if p["lang"] == lang]
    return pairs


def load_unembedding(model_id: str = MODEL_ID, safetensors_path: str | None = None,
                     revision: str | None = None) -> tuple[np.ndarray, str]:
    """unembedding 行列 W (vocab_size x hidden) と、使ったテンソル名を返す。"""
    from safetensors import safe_open

    if safetensors_path is None:
        from huggingface_hub import hf_hub_download
        safetensors_path = hf_hub_download(model_id, "model.safetensors", revision=revision)

    with safe_open(safetensors_path, framework="numpy") as f:
        keys = set(f.keys())
        for k in UNEMBED_KEYS:
            if k in keys:
                W = f.get_tensor(k).astype(np.float32)  # 必要なテンソルだけ読む
                return W, k
    raise KeyError(f"unembedding に当たるテンソルが見つかりません: 候補={UNEMBED_KEYS}")


def verify_ids(pairs: list[dict], model_id: str = MODEL_ID) -> None:
    """JSON 中の token 文字列と ID が tokenizer と一致するか確認する。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    bad = []
    for p in pairs:
        for side in ("male", "female"):
            t, i = p[side], p[f"{side}_id"]
            if tok.convert_ids_to_tokens(i) != t:
                bad.append((t, i, tok.convert_ids_to_tokens(i)))
    if bad:
        raise ValueError(f"token と ID の不一致: {bad[:10]} ... (計 {len(bad)} 件)")
    print(f"[verify] {len(pairs)} ペアの token/ID はすべて tokenizer と一致")


def compute_differences(W: np.ndarray, pairs: list[dict]) -> dict[str, np.ndarray]:
    male_ids = np.array([p["male_id"] for p in pairs], dtype=np.int64)
    female_ids = np.array([p["female_id"] for p in pairs], dtype=np.int64)

    U_m = W[male_ids]          # (N, d)
    U_f = W[female_ids]        # (N, d)
    D = U_m - U_f              # (N, d)  差ベクトル male - female

    eps = 1e-12
    diff_norm = np.linalg.norm(D, axis=1)
    D_unit = D / (diff_norm[:, None] + eps)

    # ペア内の 2 ベクトルのコサイン類似度
    pair_cos = np.sum(U_m * U_f, axis=1) / (
        np.linalg.norm(U_m, axis=1) * np.linalg.norm(U_f, axis=1) + eps)

    # 平均差ベクトル（単純平均と、単位化してからの平均）
    mean_diff = D.mean(axis=0)
    mean_unit_diff = D_unit.mean(axis=0)
    g_mean = mean_unit_diff / (np.linalg.norm(mean_unit_diff) + eps)

    # 各差ベクトルと平均方向とのコサイン（方向がどれだけ揃っているか）
    cos_to_mean = D_unit @ g_mean

    # 差ベクトルの主成分（Bolukbasi et al. 2016 風の gender direction）
    # 中心化せず SVD するとペア差の共通方向、中心化すると分散の主方向になる。
    _, S, Vt = np.linalg.svd(D_unit, full_matrices=False)
    g_pc1 = Vt[0] * np.sign(Vt[0] @ g_mean)  # 平均方向と向きを揃える
    explained = S**2 / np.sum(S**2)

    return dict(
        male_ids=male_ids, female_ids=female_ids,
        U_male=U_m, U_female=U_f, diff=D, diff_unit=D_unit,
        diff_norm=diff_norm, pair_cos=pair_cos,
        mean_diff=mean_diff, gender_dir_mean=g_mean, gender_dir_pc1=g_pc1,
        cos_to_mean=cos_to_mean, svd_explained=explained,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=PAIRS_JSON, help="男女ペアの JSON")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=None, help="HF のリビジョン（再現性のため固定推奨）")
    ap.add_argument("--safetensors", default=None, help="ローカルの model.safetensors（指定時はダウンロードしない）")
    ap.add_argument("--strict-only", action="store_true", help="strict=True のペアのみ使う")
    ap.add_argument("--lang", choices=["ja", "en", "de"], default=None, help="言語で絞り込む")
    ap.add_argument("--no-verify", action="store_true", help="tokenizer による token/ID 検証を省略")
    ap.add_argument("--out", default="ruri-v3-310m_gender_pairs_diff.npz")
    ap.add_argument("--top", type=int, default=15, help="要約で表示する件数")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs, args.strict_only, args.lang)
    print(f"[pairs] {len(pairs)} ペア (strict_only={args.strict_only}, lang={args.lang})")

    if not args.no_verify:
        verify_ids(pairs, args.model)

    W, key = load_unembedding(args.model, args.safetensors, args.revision)
    print(f"[unembedding] '{key}'  shape={W.shape}"
          + ("  (tok_embeddings と重み共有)" if "tok_embeddings" in key else ""))

    r = compute_differences(W, pairs)

    # 保存（トークン文字列などのメタ情報も一緒に）
    np.savez(
        args.out, **r,
        male_tokens=np.array([p["male"] for p in pairs]),
        female_tokens=np.array([p["female"] for p in pairs]),
        lang=np.array([p["lang"] for p in pairs]),
        category=np.array([p["category"] for p in pairs]),
        strict=np.array([p["strict"] for p in pairs]),
        unembedding_key=np.array(key), model=np.array(args.model),
    )
    print(f"[saved] {args.out}  diff.shape={r['diff'].shape}")

    # 要約
    print(f"\nSVD 寄与率（上位5）: {np.round(r['svd_explained'][:5], 3)}")
    print(f"平均方向と PC1 のコサイン: {float(r['gender_dir_mean'] @ r['gender_dir_pc1']):.3f}")
    order = np.argsort(-r["cos_to_mean"])
    fmt = "{:>8.3f} {:>8.3f} {:>8.3f}  {} / {}"
    print(f"\n平均方向に最も揃っているペア（cos_to_mean 上位 {args.top}）")
    print(f"{'cos_mean':>8} {'|diff|':>8} {'pair_cos':>8}  male / female")
    for i in order[:args.top]:
        print(fmt.format(r["cos_to_mean"][i], r["diff_norm"][i], r["pair_cos"][i],
                         pairs[i]["male"], pairs[i]["female"]))
    print(f"\n平均方向から最も外れているペア（下位 {args.top}）")
    for i in order[::-1][:args.top]:
        print(fmt.format(r["cos_to_mean"][i], r["diff_norm"][i], r["pair_cos"][i],
                         pairs[i]["male"], pairs[i]["female"]))


if __name__ == "__main__":
    main()
