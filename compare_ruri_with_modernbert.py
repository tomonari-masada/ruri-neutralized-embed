import importlib, numpy as np
m = importlib.import_module("ruri-v3-310m_gender_pairs")
Wr, _ = m.load_unembedding("cl-nagoya/ruri-v3-310m")
Wm, _ = m.load_unembedding("sbintuitions/modernbert-ja-310m")
print("相対変化:", np.linalg.norm(Wr - Wm) / np.linalg.norm(Wm))
cos = (Wr * Wm).sum(1) / (np.linalg.norm(Wr, axis=1) * np.linalg.norm(Wm, axis=1))
print("行ごとのcos 最小/1%点/中央値:", np.percentile(cos, [0, 1, 50]))

dW = Wr - Wm
print("行ノルム比 (中央値):", np.median(np.linalg.norm(Wr,axis=1)/np.linalg.norm(Wm,axis=1)))
print("平均ずれの割合:", np.linalg.norm(dW.mean(0)) / np.sqrt((dW**2).sum(1).mean()))
U, S, Vt = np.linalg.svd(dW - 0, full_matrices=False)  # 102400x768 なので数十秒〜数分
print("ΔW の寄与率 上位5:", np.round(S[:5]**2 / (S**2).sum(), 3))
g = np.load("ruri-v3-310m_gender_pairs_diff.npz")["gender_dir_pc1"]
print("性別方向と ΔW 上位特異ベクトルの |cos|:", np.round(np.abs(Vt[:10] @ g), 3))


import json
g = g / np.linalg.norm(g)
print("ΔW の g 方向エネルギー比 (1=平均的):", 768 * np.linalg.norm(dW @ g)**2 / (dW**2).sum())
pr = json.load(open("ruri-v3-310m_gender_pairs.json"))["pairs"]
mi = np.array([p["male_id"] for p in pr]); fi = np.array([p["female_id"] for p in pr])
for name, W in [("modernbert", Wm), ("ruri", Wr)]:
    print(name, "ペア差の g 成分 平均:", ((W[mi] - W[fi]) @ g).mean())
    print(name, "ペア差の g 成分 標準偏差:", ((W[mi] - W[fi]) @ g).std())

from huggingface_hub import hf_hub_download
voc = json.load(open(hf_hub_download("cl-nagoya/ruri-v3-310m", "tokenizer.json")))["model"]["vocab"]
toks = np.array([t for t, _ in voc]); score = np.array([s for _, s in voc])

# (i) W 自体の g 方向分散比。これが 1.9 前後なら、ΔW の 1.9 は異方性で説明できる
Wc = Wm - Wm.mean(0)
print("W の g 方向分散比:", 768 * np.linalg.norm(Wc @ g)**2 / (Wc**2).sum())

# (ii) ペア外の語彙で性別成分が強まったか弱まったか
pm, pr_ = Wm @ g, Wr @ g
mask = (score < 0); mask[mi] = False; mask[fi] = False
print("射影の相関:", np.corrcoef(pm[mask], pr_[mask])[0, 1])
print("ペア外 |g成分| 平均  modernbert→ruri:", np.abs(pm[mask]).mean(), "→", np.abs(pr_[mask]).mean())

# (iii) g 成分が大きく動いた語（負=女性側へ、正=男性側へ）
d = (pr_ - pm); idx = np.where(mask)[0]; o = idx[np.argsort(d[idx])]
print("女性側へ動いた:", " ".join(toks[o[:30]]))
print("男性側へ動いた:", " ".join(toks[o[::-1][:30]]))

d_ = d[mask]; p0 = pm[mask]
print("元の射影と変化の相関:", np.corrcoef(p0, d_)[0, 1], " 回帰係数:", np.polyfit(p0, d_, 1)[0])
print("変化の大きさ(最大) / 元の射影の標準偏差:", np.abs(d_).max() / p0.std())
for i in list(o[:10]) + list(o[::-1][:10]):
    print(f"{toks[i]:>14} {pm[i]:+.4f} → {pr_[i]:+.4f}")
# 「たくさん動いた」ではなく「性別方向に特化して動いた」語: ΔW_i と g の cos
c = (dW @ g) / (np.linalg.norm(dW, axis=1) + 1e-12)
oc = idx[np.argsort(c[idx])]
print("g に沿って女性側へ:", " ".join(toks[oc[:30]]))
print("g に沿って男性側へ:", " ".join(toks[oc[::-1][:30]]))