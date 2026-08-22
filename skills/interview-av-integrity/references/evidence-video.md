# 比較動画

## ケース選定

閉鎖欠如は、事前基準を満たす全件を含める。曖昧例や音素不確実例を都合よく除外せず、除外台帳へ理由を残す。

参照は同じ候補者の `visible_contact=yes` を優先し、時間帯を分散させる決定論的規則で2–3件選ぶ。必要なら他話者参照を別章にする。全ケースはclassificationではなくsource時刻順に並べる。

## レイアウト承認

レイアウトの出発点は `assets/layout-references/av-integrity-closure-review-approved.png` とする。
このPNGはユーザー提供の合成layout referenceで、実案件由来の人物や証拠を含まない。画像内の文字、人物、値はダミーであり、構造、情報階層、視線誘導、可読性、視覚品質だけをauthoritativeとする。詳細は[layout-quality-gate.md](layout-quality-gate.md)を読む。

別のImageGen conceptを初期案にしない。`scripts/render_layout_preview.py` で、実録画、原フレーム、口元crop、氏名を使わず、本番 `render_evidence_frame` 経路由来のimplementation previewを作る。scriptが作るside-by-side review sheetで参照とpreviewを比較し、previewの明示承認をユーザーから得る。参照PNGの存在や過去の承認を、不一致なimplementation previewの承認とみなさない。

明示承認後、input event manifestの `layout_review`へAPPROVED status、approval basis、preview/provenance/review-sheetのpathとhash、固定checklistの全PASSを記録する。full-render CLIは、asset、manifest basis、renderer source、preview、provenance、review sheet、approvalのどれか一つでも不一致または未完了なら、案件媒体を読む前にFAILする。変更要望や構造上の差があれば、承認を求める前にrendererを更新してpreviewを再生成する。

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
