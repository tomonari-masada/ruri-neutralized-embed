#!/usr/bin/env python3
"""
ruri-v3-310m の入力埋め込み（tok_embeddings）だけに介入し、任意のテキストの文埋め込みを求める。

介入の内容（Bolukbasi 型の中和）
  性別語（男女ペアのトークンと、「女」「母」「彼」などを含む語）と特殊トークン以外の
  すべての行 w から、性別方向 g の成分を除く:   w ← w − α (w·g) g
  g は ruri-v3-310m 自身の入力埋め込みから、男女ペアの差ベクトルの第1主成分として推定する。
  ruri-v3-310m には出力層がないので、これは「入力側だけの介入」になる。
  Transformer 本体・プーリングは一切変えない。

必要なファイル（同じフォルダに置く）
  ruri-v3-310m_gender_pairs.json
  ruri-v3-310m_gender_pairs.py              （g の推定）
  modernbert-ja-310m_gender_debias_eval.py  （残す性別語の判定）

使い方（コマンドライン）
  pip install sentence-transformers torch numpy safetensors huggingface_hub
  python ruri-v3-310m_neutralized_embed.py --texts "彼女は看護師です。" "彼は看護師です。"
  python ruri-v3-310m_neutralized_embed.py --input texts.txt --prefix "検索文書: " --out emb.npz

使い方（Python から）
  import importlib
  R = importlib.import_module("ruri-v3-310m_neutralized_embed")
  ruri = R.RuriInputNeutralizer()
  e_neu = ruri.encode(["彼女は看護師です。"], neutralize=True)
  e_org = ruri.encode(["彼女は看護師です。"], neutralize=False)

※ Ruri v3 は用途に応じてテキストの先頭にプレフィックスを付ける
  （「検索クエリ: 」「検索文書: 」「トピック: 」、意味的類似度では空文字列）。
  詳細はモデルカードを確認のこと。--prefix / prefix= で指定できる。
"""
from __future__ import annotations

import argparse
import contextlib
import importlib

import numpy as np
import torch

MODEL_ID = "cl-nagoya/ruri-v3-310m"
PAIRS_JSON = "ruri-v3-310m_gender_pairs.json"


class RuriInputNeutralizer:
    def __init__(self, model_id: str = MODEL_ID, pairs_json: str = PAIRS_JSON,
                 strict_only: bool = True, alpha: float = 1.0, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.tok = self.model.tokenizer
        self.emb = self.model[0].auto_model.get_input_embeddings().weight     # (102400, 768)

        # 性別方向 g（正 = 男性側）を ruri 自身の入力埋め込みから推定
        P = importlib.import_module("ruri-v3-310m_gender_pairs")
        self.pairs = P.load_pairs(pairs_json, strict_only=strict_only)
        W = self.emb.detach().cpu().float().numpy()
        g = P.compute_differences(W, self.pairs)["gender_dir_pc1"]
        self.g = g / np.linalg.norm(g)

        # g 成分を残すトークン（性別語・特殊トークン）
        E = importlib.import_module("modernbert-ja-310m_gender_debias_eval")
        self.keep = E.keep_ids(self.tok, self.pairs)

        # 元の行列と中和した行列を用意しておく
        self.W0 = self.emb.detach().clone()
        gt = torch.tensor(self.g, dtype=self.W0.dtype, device=self.W0.device)
        mask = torch.ones(self.W0.shape[0], dtype=torch.bool, device=self.W0.device)
        mask[torch.tensor(self.keep, device=self.W0.device)] = False         # True = 中和する行
        self.Wn = self.W0.clone()
        self.Wn[mask] -= alpha * (self.Wn[mask] @ gt)[:, None] * gt
        self.alpha = alpha
        self.n_neutralized = int(mask.sum())

    @contextlib.contextmanager
    def neutralized(self):
        """この中では中和した入力埋め込みが使われる。抜けると元に戻る。"""
        with torch.no_grad():
            self.emb.copy_(self.Wn)
        try:
            yield
        finally:
            with torch.no_grad():
                self.emb.copy_(self.W0)

    def encode(self, texts: list[str], neutralize: bool = True, prefix: str = "", **kwargs) -> np.ndarray:
        """文埋め込み (N, 768) を返す。kwargs は SentenceTransformer.encode にそのまま渡す
        （batch_size, normalize_embeddings など）。"""
        texts = [prefix + t for t in texts]
        ctx = self.neutralized() if neutralize else contextlib.nullcontext()
        with ctx:
            return self.model.encode(texts, convert_to_numpy=True, **kwargs)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", nargs="*", default=None, help="埋め込むテキスト（複数可）")
    ap.add_argument("--input", default=None, help="1 行 1 テキストのファイル")
    ap.add_argument("--prefix", default="", help='例: "検索クエリ: " / "検索文書: " / "トピック: "')
    ap.add_argument("--alpha", type=float, default=1.0, help="g 成分を除く割合（1 で完全に除去）")
    ap.add_argument("--all-pairs", action="store_true", help="g の推定に strict=False のペアも使う")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--pairs", default=PAIRS_JSON)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true", help="埋め込みを L2 正規化して保存する")
    ap.add_argument("--out", default="ruri-v3-310m_neutralized_embeddings.npz")
    args = ap.parse_args()

    texts = list(args.texts or [])
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            texts += [line.rstrip("\n") for line in f if line.strip()]
    if not texts:
        texts = ["彼女は看護師です。", "彼は看護師です。", "彼女はエンジニアです。", "彼はエンジニアです。"]
        print("[info] --texts / --input が無いので例文を使います")

    ruri = RuriInputNeutralizer(args.model, args.pairs, strict_only=not args.all_pairs,
                                alpha=args.alpha, device=args.device)
    print(f"[g] {len(ruri.pairs)} ペアから推定 | 中和した行 {ruri.n_neutralized} / {ruri.W0.shape[0]} "
          f"| 残した性別語・特殊トークン {len(ruri.keep)} | alpha={args.alpha}")

    kw = dict(batch_size=args.batch_size, normalize_embeddings=args.normalize)
    e_org = ruri.encode(texts, neutralize=False, prefix=args.prefix, **kw)
    e_neu = ruri.encode(texts, neutralize=True, prefix=args.prefix, **kw)
    self_cos = np.sum(_cos(e_org, e_neu) * np.eye(len(texts)), axis=1)

    np.savez(args.out, texts=np.array(texts), prefix=np.array(args.prefix),
             emb_original=e_org, emb_neutralized=e_neu, cos_original_vs_neutralized=self_cos,
             gender_direction=ruri.g, alpha=np.array(args.alpha), model=np.array(args.model))
    print(f"[saved] {args.out}  shape={e_neu.shape}")

    print("\n各テキストの cos(元の埋め込み, 中和後の埋め込み)")
    for t, c in zip(texts, self_cos):
        print(f"  {c:.4f}  {t}")
    if len(texts) <= 10:
        np.set_printoptions(precision=4, suppress=True, linewidth=150)
        print("\nテキスト間の cos 類似度（元）\n", _cos(e_org, e_org))
        print("テキスト間の cos 類似度（中和後）\n", _cos(e_neu, e_neu))


if __name__ == "__main__":
    main()
