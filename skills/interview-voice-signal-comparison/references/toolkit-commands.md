# Toolkit commands

以下の例では、`VOICE_SKILL_DIR`をこの`SKILL.md`があるディレクトリ、`VOICE_PYTHON`を依存物が入ったPython 3.11以上の実行ファイルに置き換える。
入力・出力pathに空白や日本語がある前提で、pathは常に引用する。

## 1. 環境確認

```bash
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/check_environment.py"
```

`status`が`PASS`で、PyAV、Matplotlib、NumPy、Pillow、SciPy、`libx264`、AAC encoderがすべて`PASS`であることを確認する。
不足がある場合は媒体を処理せず、依存物の準備についてユーザーへ説明する。

## 2. Clip manifestを固定する

[manifest-schema.md](manifest-schema.md)のschema v1で作成する。
指定群は1 clip、比較群は1〜3群、各比較群は1 clip以上とする。
`designated_group_id`と`comparison_group_ids`を必ず明記する。
再生順は`clips`配列の順なので、既定では指定clipを先頭にし、その後に比較clipを置く。

manifestを固定したら、そのSHA-256を記録する。

```bash
shasum -a 256 "/path/to/clip_manifest.json"
```

## 3. 音響特徴と指定中心比較を生成する

出力先は空または未作成のディレクトリを指定する。
`--comparison-group`は比較群ごとに1回、最大3回指定する。

```bash
MPLCONFIGDIR="/tmp/voice-signal-comparison-mpl" \
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/run_acoustic_analysis.py" \
  --manifest "/path/to/clip_manifest.json" \
  --output-dir "/path/to/analysis_output" \
  --designated-group "anchor-id" \
  --comparison-group "reference-a" \
  --comparison-group "reference-b" \
  --sample-rate 48000 \
  --include-nearest-median
```

`--include-nearest-median`は指標ごとの固有単位における最近medianを付けるだけである。
指標横断のscoreや順位は生成しない。

主な出力:

- `analysis_output/run_manifest.json`
- `analysis_output/acoustic_features/acoustic_features.json`
- `analysis_output/acoustic_features/artifact_manifest.json`
- `analysis_output/designated_centered/designated_centered_comparison.json`
- `analysis_output/designated_centered/METHOD_NOTE.md`

`acoustic_features.json`、両artifact manifest、`run_manifest.json`には、feature configに加えて実装source hash bundleとPython・主要依存物versionが記録される。

単独で再検証する場合:

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" \
  "$VOICE_SKILL_DIR/scripts/toolkit/acoustics/verify_artifacts.py" \
  --acoustic-dir "/path/to/analysis_output/acoustic_features" \
  --designated-dir "/path/to/analysis_output/designated_centered"
```

## 4. Render manifestを固定する

[manifest-schema.md](manifest-schema.md)のschema v2で作成する。
次の参照は、直前の解析で生成した同一ディレクトリの成果物を指す。

```json
{
  "feature_artifacts": {
    "artifact_manifest": {
      "path": "/path/to/analysis_output/acoustic_features/artifact_manifest.json",
      "sha256": "<actual sha256>"
    },
    "acoustic_features": {
      "path": "/path/to/analysis_output/acoustic_features/acoustic_features.json",
      "sha256": "<actual sha256>"
    }
  }
}
```

次の4値は同じ任意group IDにする。

- clip manifestの`designated_group_id`
- `comparison.anchor_group`
- `comparison.display_anchor_group`
- `comparison.point_group`

`comparison.reference_groups`はclip manifestの`comparison_group_ids`と同じ順序にする。
`delta_definition`は正確に`group metric minus user-designated point metric`とする。

## 5. 動画とmappingを生成する

本番の前に、実録画を外部送信せずローカルでpreviewを作る。
previewは実際のgroup名・色・指標rangeを使うが、波形とSTFTは合成placeholderであり、その旨を画像内に表示する。

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/render_layout_preview.py" \
  --manifest "/path/to/render_manifest.json" \
  --output "/path/to/output/layout_preview.png"
```

previewをユーザーに示し、既定レイアウトから大きく変える場合は承認を得る。
承認後に本番動画を生成する。

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/render_comparison.py" \
  --manifest "/path/to/render_manifest.json" \
  --output "/path/to/output/comparison.mp4" \
  --effective-manifest "/path/to/output/effective_manifest.json" \
  --frame-map "/path/to/output/frame_map.csv" \
  --audio-map "/path/to/output/audio_map.csv"
```

rendererは入力manifestを上書きせず、source、interval、authoritative summary、artifact hashを再検証する。

## 6. 動画をfail-closedで検証する

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/verify_comparison.py" \
  --video "/path/to/output/comparison.mp4" \
  --effective-manifest "/path/to/output/effective_manifest.json" \
  --frame-map "/path/to/output/frame_map.csv" \
  --audio-map "/path/to/output/audio_map.csv" \
  --output-dir "/path/to/output/qa"
```

終了コード0だけで完成扱いにしない。
`qa/comparison_verification.json`の`status`が`PASS`であることを確認し、`qa/VISUAL_QA_contact_sheet.png`を人間が確認する。
機械QAがPASSでも、視覚・聴取QAが未完ならその状態を明記する。

## 7. 開発時の回帰テスト

```bash
PYTHONPATH="$VOICE_SKILL_DIR/scripts/toolkit" \
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" -m unittest discover \
  -s "$VOICE_SKILL_DIR/scripts/toolkit/acoustics/tests" -v

PYTHONPATH="$VOICE_SKILL_DIR/scripts/toolkit:$VOICE_SKILL_DIR/scripts/toolkit/rendering" \
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" -m unittest discover \
  -s "$VOICE_SKILL_DIR/scripts/toolkit/rendering/tests" -v
```

テスト後、skill package内の`__pycache__`と`.pyc`を納品物へ残さない。
