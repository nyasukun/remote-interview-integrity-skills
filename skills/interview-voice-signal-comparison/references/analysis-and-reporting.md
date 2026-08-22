# 音響分析と報告

## Feature contract

全clipへ同じ設定を適用する。

- waveform: 48 kHz解析音声のmono平均。表示用の局所scaleと絶対RMSを区別する。
- Log-Mel: 同じFFT、window、hop、mel filter、dB変換を使う。
- F0: 有声frameだけの自己相関系推定。探索範囲と欠測条件を記録する。
- periodicity: F0候補lagの正規化自己相関peak。
- activity: 短時間RMSに基づくactive-frame fraction。
- voiced fraction: active frameのうちF0を採用できた比率。
- RMS dBFS: playback音量ではなくsource区間の記述値。
- spectral centroid、flatness、band fractions。
- stereo balance、correlation、side fractionは収録channel診断。dual-monoなら非識別的と明記する。

feature configと実装hashを成果物へ残す。

## Group summary

群ごとにclip単位のscalar summaryを無加重で集約する。

- `value_count`
- `min`
- `max`
- `median`

clip数が1なら`min = median = max`となり、rangeではなくpointである。
発話frameを全部poolした値と、clip summaryの中央値を混同しない。

## 指定中心比較

指標`m`、比較群`g`について次を計算できる。

```text
signed_delta(g,m) = median(g,m) - value(designated,m)
inside(g,m) = min(g,m) <= value(designated,m) <= max(g,m)
```

必要なら、その指標の固有単位で`abs(signed_delta)`が最小のmedianを「この指標で最も近い」と記述する。
tiesはすべて残す。

## 合算禁止

次を作らない。

- 指標横断の平均距離
- z-scoreやmin-max正規化後の総合score
- 多数決によるclosest speaker
- speaker similarity、match probability、same-speaker probability
- embedding distanceやvoiceprint

理由:

- Hz、比率、dBFS、スペクトル比は単位・尺度が異なる。
- 任意の標準化と重みで順位が変わる。
- F0、periodicity、voiced fraction、帯域比には相関や構成比の依存がある。
- 発話内容、言語、プロソディ、マイク、AGC、noise suppression、codec、会議mixが交絡する。
- 通常は各群1〜3 clipで、本人内分布を推定する標本ではない。

## 報告順序

1. **観測条件:** clip数、長さ、label origin、録音経路。
2. **指定値:** 指定clipの各metric。
3. **指標別比較:** range包含、符号付き差、必要なら指標別最近median。
4. **不一致:** ある指標では群A、別の指標では群Bが近いことをそのまま示す。
5. **限界:** 内容・言語・収録差、mixed track、sample size。
6. **次の確認:** 同一文、同一端末、同一距離、背景処理off等の制御録音。

## 推奨表現

使う:

- 「F0中央値では指定値は参照Aの範囲内だった」
- 「周期性中央値では参照Bのmedianが指定値に最も近かった」
- 「活動率と帯域比で近い群が一致しなかった」
- 「これは指標別の記述であり、話者の同一性を示さない」

避ける:

- 「総合的に候補者本人の声である」
- 「面談者Aとは別人である」
- 「この声は北朝鮮・韓国・日本の話者である」
- 「偽装、代理発声、詐欺が確定した」
- 「採用・契約すべきでない」と音声比較だけから結論する

## 採用・契約との接続

本比較を単独の採否根拠にしない。
必要なら全候補へ同じ手順で、ランダム文のライブ読み上げ、別端末・別回線での再面談、画面共有付き課題、適法な本人・契約主体確認を行う。
