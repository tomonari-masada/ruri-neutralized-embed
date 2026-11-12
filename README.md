# debias ruri-v3-310m
* cl-nagoya/ruri-v3-310mのinput embeddingをdebiasすることで埋め込みを変化させる

## 準備
```sh
python ruri-v3-310m_gender_pairs.py
python ruri-v3-310m_gender_pairs.py --top 100
python ruri-v3-310m_gender_pairs_checks.py
python ruri-v3-310m_gender_pairs.py --top 100
python compare_ruri_with_modernbert.py 
python modernbert-ja-310m_gender_debias_eval.py 
```

## debiasのチェック
* 「彼女は看護師です」と「彼女はエンジニアです」の類似度の差と、「彼は看護師です」と「彼はエンジニアです」の類似度の差が、介入後に縮まるかどうかを見る。
  ** 男性側の例は、エンジニアより、消防士・大工・警察官 のように職業→性別で強く男性に寄っていた職業のほうが向いているかもしれない。
```sh
python ruri-v3-310m_neutralized_embed.py --texts "彼女は看護師です。" "彼女はエンジニアです。"
python ruri-v3-310m_neutralized_embed.py --texts "彼は看護師です。" "彼はエンジニアです。"
```
