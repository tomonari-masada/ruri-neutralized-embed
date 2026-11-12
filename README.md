# ruri-neutralized-embed
cl-nagoya/ruri-v3-310mのinput embeddingをdebiasすることで埋め込みを変化させる

```sh
python ruri-v3-310m_gender_pairs.py
python ruri-v3-310m_gender_pairs.py --top 100
python ruri-v3-310m_gender_pairs_checks.py
python ruri-v3-310m_gender_pairs.py --top 100
python compare_ruri_with_modernbert.py 
python modernbert-ja-310m_gender_debias_eval.py 
```
