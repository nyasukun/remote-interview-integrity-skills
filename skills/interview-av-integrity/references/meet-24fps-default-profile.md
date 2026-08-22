# Meet系24fps参考profile

このprofileは、再利用可能な未校正の初期値を整理したものである。
ベンチマーク、一般的な検出性能、誤検出率を示す「検証済み閾値」ではない。
Google Meet系のspeaker-view合成録画、1920×1080、24fps CFR、AAC 48kHzでの候補値としてのみ使う。
fps、解像度、codec、タイル寸法、話者が違う場合は、positive controlと目視盲検で案件ごとに校正する。

## 音響

- decode: 48 kHz
- STFT: 8 ms Hann、1 ms hop、FFT 1024
- release候補: energy rise、12 ms slope、高域rise、spectral flux
- 粗アンカー探索: おおむね -300〜+220 ms。語・発音により拡張する
- top-1の無条件採用は禁止。保守ゲート不通過は音響のみでtop候補をレビュー

## 口唇

- inner-lip central pairsを口角局所軸へ投影し、口幅で正規化
- 3点対のmedianとspreadを品質監査
- 反復frameを連続接触の票に数えない
- 極端なyaw/pitch、口幅不足、遮蔽、ランドマーク急跳びを除外

初期値の目安:

- contact candidate: median aperture <= 0.015かつmaximum <= 0.025
- reopen: median aperture >= 0.035
- 3点対spread > 0.020は慎重に扱う
- 接触・再開放は2つ以上のnon-repeated frameで確認

この数値だけで閉鎖欠如を確定しない。
人物ごとの唇形状、髭、解像度で校正し、blind native-frame reviewを併用する。

## 時間分解能

```text
frame_period_ms = 1000 / fps
```

24fpsでは41.67ms。通常、1–2 frame未満の差を強く解釈しない。
slowは各native frameを4回保持して0.25xとし、補間しない。

口唇音響相関のparticipant別offsetは、WebRTC、jitter buffer、会議合成、録画muxで変わり得る。
候補者に安定したoffsetがあっても、口パクを特異的に示さない。
