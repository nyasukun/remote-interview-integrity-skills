# 比較動画

## ケース選定

閉鎖欠如は、事前基準を満たす全件を含める。曖昧例や音素不確実例を都合よく除外せず、除外台帳へ理由を残す。

参照は同じ候補者の `visible_contact=yes` を優先し、時間帯を分散させる決定論的規則で2–3件選ぶ。必要なら他話者参照を別章にする。全ケースはclassificationではなくsource時刻順に並べる。

## レイアウト承認

[layout-quality-gate.md](layout-quality-gate.md)を読み、同梱参照画像の確認、本番renderer由来の合成preview、side-by-side review、ユーザーの明示承認を完了する。
承認記録は[artifact-schemas.md](artifact-schemas.md#layout_review)の `layout_review` に固定する。hard gateがPASSするまでfull renderを開始しない。

## 既定レイアウト

- 同梱rendererは24fps CFR入力専用。非24fps・VFRを黙って変換せず、原フレーム1:1を保つ別方法を合意できなければ停止する
- 1920×1080、24fps CFR、H.264 yuv420p、AAC 48 kHz
- 参照PNGは1672×941だが、implementation previewと本番出力は1920×1080のままとする
- 上段: 原映像由来の同一source frame
- 右上: 同じsource frameから切り出した口元inset
- 下段: stereo平均のRMS envelope、固定release marker、移動playhead
- ケースごとにnormalと0.25x slow
- slowは各native frameを厳密に4回holdし、光学補間しない
- slow audioは同じsource windowを4倍へ線形伸長。pitch非保持をmanifestへ記録
- intro/outroは最低5秒を目安にし、限定文を読める長さにする

24fpsで±0.8秒を要求しても1.6秒は整数frameにならない。normalの1:1対応を優先し、実効窓をwhole frameへ量子化し、requested/effectiveをmanifestで分ける。

## 表示規則

- 欠如: 「可視フレーム内で明瞭な閉鎖を確認できず」
- 参照: 「閉鎖あり参照」。基準を満たす場合だけ「同期参照」
- release時刻、event ID、対象語/音素、通常/低速、フレーム不確実性
- 遅い接触は「後続音素の口形である可能性あり」
- 語彙意図と実現音が違う場合は黄色等の中立的な注意表示
- ケースごとに波形を正規化するなら、絶対音圧比較不可と明示

intro/outroの定型文:

> 本資料は、録画内の音響開放時刻と可視的な口唇接触を並べた観測資料です。A/V不整合の原因、発話者の本人性、国籍、所属、意図を判定するものではありません。

## frame map

各output frameについて次を保存する。

- `output_frame_index`, `output_pts_s`
- `case_order`, `event_id`, `phase`
- `source_pts_s`, `source_frame_index`

normalは連続native framesを1:1、slowはnormalと同じsource index列を各4回保持する。カードやgapはsource列を空にする。
