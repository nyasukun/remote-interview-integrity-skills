# 解析プロトコル

## 1. 事前固定

解析前に次を `analysis_protocol.md` へ記録する。

- 原動画のSHA-256と解析対象の絶対パス
- 対象開始・終了時刻
- 候補者と対照者の割当根拠
- 主解析 `/p/`、副解析 `/b/`
- ASR信頼度、話者切替guard、顔品質、口幅、姿勢、遮蔽の除外規則
- 音響開放の探索窓と保守ゲート
- 口唇閉鎖窓、再開放窓、フレーム不確実性
- 主要評価量と感度解析
- 参照例の選定規則

結果を見た後の変更は、時刻、理由、影響範囲をmethod amendmentへ追記する。元規則を削除しない。

## 2. 原本と時間軸

コンテナのpresentation timestampを基準にする。デコード順や `frame_index / nominal_fps` だけで時刻を作らない。

記録項目:

- video/audio codec、解像度、time base、start PTS、duration
- average/nominal fps、CFR/VFR、重複・freeze、欠損PTS
- sample rate、channels、A/V終端差
- 会議サービスによるactive-speaker切替と合成レイアウト

VFRや不連続PTSでは、24fps向け閾値を流用しない。すべて秒または実PTSで評価する。

## 3. 話者区間と同期エポック

表示名、タイル位置、会話ターンを用い、顔貌・声紋・訛りで本人を割り当てない。表示切替の前後にguardを置く。レイアウト、解像度、入力端末、回線状態が変わる区間は別epochにする。

最低限の列:

- `start_s`, `end_s`, `speaker`, `group`, `epoch_id`
- `assignment_signal`, `confidence`, `exclusion_reason`

候補者は `group=candidate`、他話者は `group=control`、不確実区間は `group=exclude` とする。

## 4. token inventory

日本語ではパ行を主解析、バ行を副解析とする。`/m/` は両唇音でも破裂音ではないため混ぜない。

表層文字のパ・バ行検索だけでは、「場所」「日本」等の漢字表記に含まれる /b,p/ を落とす。
全件台帳には、対象範囲の全wordに `reading`、`kana`、`pronunciation` のいずれかを付与し、`--require-complete-readings` で欠落0件を確認する。
読みなしの実行は探索的なliteral-kana inventoryに限定し、網羅性の主張や閉鎖欠如「全件」動画に使わない。

ASRは語と対象文字の事前選択、音響release探索の粗いアンカー、文脈による対象音素の候補提示にだけ使う。ASR開始・終了をrelease時刻にしない。各tokenは、`orthographic_class`、`intended_phone`、`acoustic_realization` を分離する。

## 5. 音響開放

映像を見ず、48 kHz原音または可逆抽出を用いる。波形だけでなく、閉鎖による低エネルギー、広帯域burst、高域エネルギー、12 ms energy slope、spectral flux、VOTを確認する。

自動候補は高特異度で採用し、曖昧なtop-1を自動確定しない。上位候補を音声のみのランダム順レビューへ回す。対象音素でない、releaseが分離不能、重複発話、ノイズ抑制で消失した場合は `unmeasurable` にする。

固定後の音響注釈には、release時刻、境界不確実性、選択rank、実現音、confidence、理由、ファイルhashを含める。

## 6. 口唇接触と可視開放

内唇の中央3点対を、口角方向の局所軸と口幅で正規化する。軽いroll/yawは局所軸で吸収し、極端な姿勢は除外する。

区別するもの:

- `visible_contact`: 高品質なnative frameで接触が見える
- `no_visible_contact`: 指定窓を十分観測したが明瞭接触がない
- `indeterminate`: 解像度、遮蔽、フリーズ、姿勢、フレーム間隠れで判断不能

自動ランドマークは候補抽出と品質監査に使う。閉鎖欠如の確定には、音声・群・機械分類を隠したnative-frame目視を併用する。

可視releaseは「接触状態から再開放した境界」である。単に口が開いているフレームをreleaseとしない。

## 7. 結合と方向

音響・視覚注釈を固定した後、event IDで結合する。

```text
release_lag_ms = (visual_release_s - acoustic_release_s) * 1000
```

- 正: 音響が先、可視開放が後
- 負: 可視開放が先、音響が後
- near: フレーム分解能と境界不確実性以内
- unavailable: 閉鎖・再開放を確定できない

閉鎖onsetとburstを比べる場合、閉鎖→burstは通常の生成順序である。burst→対応閉鎖は同一音素として不自然だが、固定A/V offset、隣接音素、ASR誤帰属でも生じるため、それだけで口パクとはしない。

## 8. 対照と統計

優先順位:

1. 同じ候補者・同じレイアウトの閉鎖あり例
2. 同じ録画・近い時刻の他話者
3. 同じcodec epoch、解像度、姿勢、発話量のmatched window

token数を有効標本数にしない。speaker/device/networkとepochが共有要因なので、epoch単位で中央値等へ集約する。候補者1人対対照2人のような設計では、token置換p値を群推論として扱わない。

最低限、件数、測定可能率、除外理由、符号付きlag、絶対lag、方向、effect size、bootstrap区間、話者別・epoch別分布を併記する。多重探索を行った場合は主解析と探索解析を分離する。
