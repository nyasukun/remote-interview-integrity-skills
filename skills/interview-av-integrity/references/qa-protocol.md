# QAプロトコル

## 機械QA

1. input event manifest、effective manifest、frame map、MP4のhashを記録する。
2. event IDが一意、release時刻が有限かつsource範囲内、ROIが0–1内、source時刻順、classification countsが一致することを検証する。
3. MP4を全編decodeし、video/audio各1stream、codec、1920×1080、24fps CFR、yuv420p、AAC 48kHz、frame count、duration、A/V終端差を検証する。
4. frame map行数とdecoded frame数を一致させる。
5. normalはnearest native frameの連続1:1、slowはnormalのsource列と一致し、各frameを厳密4回保持することを検証する。
6. release marker output時刻がsource releaseと速度写像から1 frame以内で一致することを検証する。
7. normal/slowの各audio phaseに実音があり、意図しないdropoutがないことを確認する。
8. 全出力のSHA-256と検証コマンドを保存する。

CLI終了コード0だけで成功としない。解析smokeでは、attempted件数、`processing_error`、measured/excluded/deferredの内訳も確認し、全attemptedがerrorならFAILにする。

## 目視QA

全ケースについてnormal/slowの開始、release、終了を確認する。

- 上段が原映像である
- insetが同じsource frameの口元である
- cropが口唇・顎を十分含み、別人やUIだけを切り出していない
- 波形、固定marker、playhead、判定色、時刻、event IDが読める
- 日本語が切れていない
- 欠如/参照の色と文言がmanifestに一致する
- 遅い接触・音素曖昧性の注記が必要箇所にある
- intro/outro限定文を読める

全normal unique ROI frameとslow reuseをcontact sheetまたは署名で監査する。QA担当は、可能ならrenderer担当と別のagentにする。

## negative tests

次を意図的に壊してFAILになることを確認する。

- duplicate event ID、逆時系列、release範囲外、ROI範囲外
- markerを100msずらす
- slow holdを3回または別source frameへ変える
- 1 phaseを無音化する
- wrong fps/resolution/sample rate、decode途中破損
- 限定文を原因・本人性・国籍・所属の断定へ変える

再利用性の最小テストは、過去のevent IDを含まないmissing/reference各1件のnovel manifestがrenderとgeneric QAを通ることである。
