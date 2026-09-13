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

`measurable` にできるのは、対象と同じ `p` または `b` の実現を音声側で確認できた行だけである。
母音の立ち上がり、隣接音節、`other`、`uncertain` を対象破裂音の時刻として採用しない。
短い破裂成分を分離できない場合は、強いエネルギーピークへ置き換えず `unmeasurable` にする。
自動候補が0件でも音声資料を作り、測定不能例を台帳に残す。

`prepare_blinded_acoustic_review.py` のschema v3 keyは、入力eventsのSHA-256、レビュー窓、復号できた音声区間 `review_window.covered_intervals` を保持する。
`finalize_blinded_acoustic_review.py` は、入力hash、event ID、anchor、候補時刻、選択時刻の窓内包含を検証する。
ASRや自動解析を再実行した場合はkeyとレビュー資料も作り直し、連番IDが同じでも古い注釈を転用しない。
`review_window` または `covered_intervals` がない旧keyは再生成が必要である。
窓内でも、PTS欠落やファイル末尾のpaddingは測定可能な音声として扱わず、その区間の時刻は採用しない。
確定後の方向・閉鎖解析でも、記録された入力hashが解析対象eventsと一致することを検証する。
旧形式の確定済み注釈で実現音が未記録の場合は `legacy_unspecified` として区別し、確認済みの対象音素と同一視しない。

### 自動候補の読み方

runnerは対象語と前後のASR word区間を検出器へ渡す。
`attribution=target` は粗い語区間との対応を示し、対象の破裂音が確認済みという意味ではない。
同じ語内の母音や他の子音の立ち上がりも競合する。両唇音が語内に1つだけ、または検出候補が1つだけでも、その候補を対象音素とは確定しない。
区間境界の競合、重複する語区間、同一語内の複数の両唇音に対する割り当てが解けない場合は保留する。
候補の表示件数を減らしても、採否判定には切り捨て前の候補を使う。

短いフレームの広帯域スペクトル証拠が乏しく、周期的な母音のエネルギー上昇だけで説明できる候補は、自動releaseの選択対象から外す。候補時刻、スコア、スペクトル証拠の測定値、除外理由は監査用に残す。
この判定の閾値は未校正の保守的ヒューリスティックであり、通過しても /p,b/ の調音位置や語内対応を証明しない。有声の /b/ は閉鎖中にも周期成分を持ち得るため、周期成分の存在だけで除外したり、直前の完全な無音を必須にしたりしない。

近いピークをまとめる `same_event_window_ms=40` と、間のエネルギー連続性を調べる `same_event_continuity_db=10` は未校正の初期値である。
`same_event_as_selected` はこの規則で同じ候補群になったことを示す。
波形の立ち上がりが破裂、気息、母音のどれに当たるかは、このフラグや `confidence=high` だけでは決まらない。
音響時刻を映像の口形へ合わせて選び直さず、特定の語だけに固定の時間補正を加えない。候補を棄却した後も、別のピークを自動的に正解とせず、音声側で対応が確定しなければ測定不能として残す。

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
