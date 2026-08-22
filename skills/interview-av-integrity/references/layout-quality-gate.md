# A/V証拠動画のレイアウト品質ゲート

## 承認済みの出発点

`assets/layout-references/av-integrity-closure-review-approved.png` を、閉鎖証拠動画のauthoritative starting layoutとする。
動画作業に入るagentは、まずこのPNGを画像として開き、`assets/layout-references/manifest.json` のhash・寸法と一致することを確認する。

画像中の架空人物、部屋、数値、ケース番号、単語、音素、波形はダミーである。
一方、次の情報階層と視覚品質は、本番rendererの出発点として扱う。

- 上部のケース数、観測分類、再生速度、赤/緑凡例
- 大きな原映像枠と、口元ROIから同一source frameの拡大insetへの視線誘導
- 口元を判定しやすいinsetサイズと境界
- 下段波形、閉鎖確認窓、破裂点強調、固定release marker、移動playhead
- 暗色背景、判定色、補助色の使い分けと、日本語の可読性

## 承認までの必須手順

1. 承認済みPNGを確認する。初期案として別のImageGen conceptを作らない。
2. `scripts/render_layout_preview.py` を使い、本番と同じ `render_evidence_frame` 経路から合成プレースホルダのimplementation previewを作る。実録画、原フレーム、口元crop、氏名を使わない。
3. scriptが生成するside-by-side review sheetで、承認済み参照とimplementation previewを比較する。
4. 参照画像自体ではなく、本番renderer由来のimplementation previewをユーザーに提示する。ユーザーがそのpreviewを明示承認するまでfull renderを始めない。
5. 明示承認後、input event manifestの `layout_review` へAPPROVED status、approval basis、preview/provenance/review-sheetのpathとSHA-256、固定checklistの全PASSを固定する。このhard gateをコードで再検証するまでfull renderを開始しない。
6. 差異がある場合は、承認を求める前にrendererを修正する。ダミー文言や案件依存ROI等のadaptableな差は、理由を説明する。

承認済み参照には人物表現があるが、本番previewへの実案件顔画像の使用を許可するものではない。

## 機械ゲート

- 承認済みassetのrelative path、SHA-256、byte size、寸法、color modeをmanifestと実ファイルから再検証する。
- implementation preview、review sheet、renderer source、`layout_review` を除いたinput-manifest basisのSHA-256をpreview provenanceへ記録する。
- full-render CLIは、`layout_review` のAPPROVED status、明示承認basis、固定checklist、preview/provenance/review-sheet hash、manifest basis、renderer source hashを案件媒体の読み込み前に再検証する。
- 本番effective manifestへ承認済みlayout referenceとlayout reviewのprovenanceを記録し、本番verifierが現在のasset、input manifest、preview artifactsと一致することを確認する。
- この機械ゲートは視覚的な良否を自動判定しない。review reportは明示承認まで `PENDING_USER_APPROVAL` とする。

スキルの設置先またはrenderer sourceが変わった場合、既存の承認記録を移植して使わない。
新しい実行環境でpreviewとreview sheetを再生成し、再承認する。

## 不合格条件

- asset、asset manifest、hash、寸法のいずれかが一致しない
- 承認済み参照を開かず、新規ImageGen mockupを初期案にする
- implementation previewが本番 `render_evidence_frame` 経路で作られていない
- previewに実案件画像、氏名、表示名が入る
- 参照とpreviewを並べた比較をせずに承認を求める
- 明示承認前にfull renderを始める
- `layout_review`がない、APPROVEDでない、固定checklistにfalse/過不足がある
- preview承認後にrenderer source、input-manifest basis、preview、provenance、review sheetのいずれかが変わる
