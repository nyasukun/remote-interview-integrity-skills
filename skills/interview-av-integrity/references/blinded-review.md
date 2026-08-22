# 盲検レビュー

## 原則

音響判定と視覚判定を、相互の結果を見ない状態で固定する。匿名ID、固定seed、ランダム順を使い、keyは判定確定後まで開かない。

同じagentの別passは「独立査読」と呼ばない。可能なら、結論や想定バグを渡さず、`fork_turns="none"` の別subagentへレビューを委ねる。顔や会話内容から話者を推測できる場合、blind qualityを `partial` と記録する。

## 音響レビュー

提示してよいもの:

- 匿名ID
- 対象音素と語の最小限の文脈
- 波形・スペクトログラム
- ASR粗アンカーを0とした相対時刻
- 自動上位候補。ただし選択済み候補や映像lagは隠す

隠すもの:

- 映像、口形、話者、candidate/control、機械の視覚分類
- 仮説、期待方向、他レビュー結果

出力enum:

- `measurable`, `unmeasurable`
- `p`, `b`, `other`, `uncertain`
- `high`, `medium`, `low`

時刻はabsoluteとanchor-relativeの両方を保存する。注釈CSVとblind keyを別々にhash化する。

## 視覚レビュー

原則として音声を消し、匿名ID、native frame、相対フレーム時刻だけを示す。最初のpassではspeaker/group、語、音響release marker、機械分類を隠す。release近傍の検索が必要な第二passでも、群と機械分類は隠す。

推奨enum:

- `visible_contact=yes|no|ambiguous`
- `first_contact=before_or_at|after_only|none|ambiguous`
- `contact_strength=strong|weak|none|ambiguous`
- `confidence=high|medium|low`

低解像度、髭、手、マイク、頭部姿勢、照明、圧縮、freezeをnoteに残す。

## 複数評価者

強い「閉鎖欠如」主張には、可能なら2–3評価者を用いる。2/3多数決だけでなく、全会一致数、不一致表、Cohen/Fleiss kappa、ambiguous率を併記する。評価者ごとに列名を変えず、同じテンプレートを使う。

多数決で `no` でも、後続500 ms程度を追跡し、遅い接触が隣接音素の口形である可能性を確認する。遅い接触を対象音素へ自動帰属しない。

## adjudication

再判定は元のblind原票を上書きしない。独立判定、不一致、再点検、最終採用理由を別ファイルで残す。語彙上の意図と音響実現が異なるときは、両方を保存する。
