# 指定中心動画とQA

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

本番前にsynthetic/local previewを作る。
実録画、顔、氏名、原音声をimage generationへ送らない。

previewで確認する。

- 指定anchorが固定される
- 参照群の色・labelが中立的
- 0線、range、median、軸domainが読める
- 「符号付き差であり、距離・順位ではない」が読める
- limitation stripが読める

既定templateから大きく変える場合、ユーザー承認後に本番renderする。

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
2. designated群が1 clip、比較群が1〜3群、clip/group IDが一意であることを検証する。
3. `anchor_group = display_anchor_group = point_group = designated_group`を検証する。
4. signed deltaの指定群が全metricで0であることを検証する。
5. effective manifest全体を再帰走査し、composite score、ranking、winner、speaker similarity、same-speaker probability、identity fieldが存在しないことを検証する。
6. MP4を全編decodeし、video/audio各1stream、codec、resolution、fps、PTS、frame数、A/V終端差を検証する。
7. frame mapの行数・PTS・phase・clip ID・group ID・source timeをdecoded videoとeffective clip区間に一致させる。
8. audio mapの連続性、source interval、各clipの実音、clip順をeffective timelineとauthoritative intervalに一致させる。
9. source clipを再decodeし、fade/gain適用後のoutput phaseと相関、lag、RMS差、normalized RMSEを比較する。
10. 全frameのlimitation stripをpixel参照で検証する。

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
