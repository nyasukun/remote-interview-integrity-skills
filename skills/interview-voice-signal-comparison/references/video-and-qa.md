# 指定中心動画とQA

## 承認済み構図基準

最初に`assets/layout-references/voice-signal-designated-comparison-approved.png`をviewし、`assets/layout-references/layout-reference-manifest.json`で固定したSHA-256、byte数、PNG/RGB、1672×941を検証する。
この画像はユーザー提供のsynthetic layout referenceであり、案件証拠ではない。

正本にするのは、上下の情報階層、左右panel、波形・Log-Mel・下段指標panelの配置、暗色の技術レビュー表現、accent color、情報密度、限定文の視認性である。
人物・silhouette、文字、言語名、値、timestamp、再生状態、波形、spectrogram、密度曲線、chart geometry、1672×941 canvasは正本にしない。
現行の指定中心rulesと1920×1080 production profileが常に優先する。

## 既定レイアウト

- 1920×1080、24 fps CFR、H.264 yuv420p、AAC stereo 48 kHz。
- 左上: 指定clipを全編固定表示。橙等のanchor色を使う。
- 右上: 現在再生中の比較clip。群ごとに色を固定する。
- 両panel: waveform、Log-Mel、clip label、長さ、playhead。
- 下段: 指定値を0とする共有軸。比較群のmin–maxを線、medianを点で表示する。
- 推奨panel: F0・periodicity、activity・voiced・RMS、low/mid/high band fraction。
- stereoが全区間dual-monoならchartから外し、非識別的という注記だけを残す。
- 全frame下端: `観測できる違いを表示。本人性は判定しない。`

波形とLog-Melをclipごとに表示scaleへ合わせる場合、それを絶対的な群間振幅比較に使わない。
RMS等の比較値はsource PCMから同じ実装で計算する。

## Preview

新しいimagegen案を作らない。
承認済み構図基準をviewした後、同梱の`render_layout_preview.py`からproduction rendererの`_clip_frame` pathを使ったsynthetic/local previewを作る。
実録画、顔、氏名、原音声をimage generationへ送らない。
previewの波形とLog-Melはplaceholderであり、production layoutの構造検査にだけ使う。

previewで確認する。

- 基準画像と同程度に、左右panel、波形、Log-Mel、下段指標、限定文の視覚階層が明瞭である
- 指定anchorが固定される
- 参照群の色・labelが中立的
- 0線、range、median、軸domainが読める
- 「符号付き差であり、距離・順位ではない」が読める
- limitation stripが読める
- dual-mono channel表示を識別情報として扱っていない
- previewがproduction renderer pathで作られている

previewと同時に、左へ基準画像、右へproduction preview、下端へ比較checkを置いた1920×720のside-by-side review sheetを自動生成する。
previewはPNGと同名の`.provenance.json`を生成し、基準asset/manifestのpath・hash・寸法、preview hash・寸法、review sheet hash・寸法、renderer source hash、render manifest basis hashを保存する。
上記checkをreview sheet上で全件確認した後にだけreview sheetをユーザーへ示して承認を求める。
明示承認後、render manifestの`layout_review`へpreview/review sheet/provenanceのpath・hash、全check、approval basisを固定する。
production rendererはこの記録が欠ける、未承認、hash不一致、check未完なら停止する。

スキルの設置先またはrenderer sourceが変わった場合、既存の承認記録を移植して使わない。
新しい実行環境でpreviewとreview sheetを再生成し、再承認する。

## Timeline

- intro/outroは限定文を読める長さにする。
- 原音声clipは重ねず、順番に再生する。
- clip間に短い無音gapを置ける。
- clip edge fadeと全体clipping防止gainをeffective manifestへ記録する。
- clipごとのloudness normalizationは既定で行わない。

## Frame map

各output frameについて保存する。

- `output_frame_index`, `output_pts_s`
- `phase`, `clip_id`, `group_id`
- 対応するsource audio時刻またはsample範囲

このvideoはsource映像を表示しないため、口元やsource video frameを生成・補間しない。

## Audio map

各phaseについて保存する。

- output sample start/end
- phase、clip ID、group ID
- source start/end
- fade、global gain

phaseは連続し、gap・overlapがないことを検査する。

## 機械QA

1. input/effective manifest、feature artifacts、source、MP4、mapsのhashを再計算する。
2. 承認済みlayout referenceのallowlist、asset/manifest hash・byte数・PNG/RGB・1672×941、production canvas 1920×1080を再検証する。
3. preview/review sheet/provenance hash、render manifest basis、renderer source hash、全quality check、明示承認記録を再検証する。
4. designated群が1 clip、比較群が1〜3群、clip/group IDが一意であることを検証する。
5. `anchor_group = display_anchor_group = point_group = designated_group`を検証する。
6. signed deltaの指定群が全metricで0であることを検証する。
7. effective manifest全体を再帰走査し、composite score、ranking、winner、speaker similarity、same-speaker probability、identity fieldが存在しないことを検証する。
8. MP4を全編decodeし、video/audio各1stream、codec、resolution、fps、PTS、frame数、A/V終端差を検証する。
9. frame mapの行数・PTS・phase・clip ID・group ID・source timeをdecoded videoとeffective clip区間に一致させる。
10. audio mapの連続性、source interval、各clipの実音、clip順をeffective timelineとauthoritative intervalに一致させる。
11. source clipを再decodeし、fade/gain適用後のoutput phaseと相関、lag、RMS差、normalized RMSEを比較する。
12. 全frameのlimitation stripをpixel参照で検証する。

CLI終了コード0だけでPASSにしない。

## 目視・聴取QA

- intro、各clipのstart/mid/end、outroを確認する。
- 指定anchorが常に左panelと0線にいる。
- 右panelのlabel、色、波形、Log-Mel、playheadがmanifestと一致する。
- 軸domain、range、median、0線がclip間で変わらない。
- 日本語と限定文が切れていない。
- winner、本人一致、国籍、所属、意図の表示がない。
- 全clipが聞き取れ、意図しないdropout・重なり・極端な音量変化がない。

可能ならrenderer担当と別のagentがfinal QAを行う。

## Negative tests

少なくとも次を壊すとFAILすることを確認する。

- anchor fieldの一つだけを比較群へ変更
- 指定群を2 clipに変更
- duplicate clip/group ID
- signed deltaの指定値を0以外へ変更
- composite scoreやwinner fieldを追加
- nested summaryへspeaker similarity、same-speaker probabilityを追加
- source hash、feature artifact hash、map PTSを変更
- frame mapのsource time、同一群内clip ID、audio mapのsource intervalを変更してhashも更新
- audio phaseを無音化またはsample gapを挿入
- limitation textを本人性断定へ変更
- layout reference asset、manifest、preview、side-by-side review sheet、preview provenanceの1 byteを変更
- `layout_review`を削除、`status`を未承認へ変更、quality checkを1件falseにする
