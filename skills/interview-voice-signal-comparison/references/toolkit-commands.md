# Toolkit commands

以下の例では、`VOICE_SKILL_DIR`をこの`SKILL.md`があるディレクトリ、`VOICE_PYTHON`を依存物が入ったPython 3.11以上の実行ファイルに置き換える。
入力・出力pathに空白や日本語がある前提で、pathは常に引用する。

数値比較は工程1〜3、動画制作は続けて工程4〜6を使う。工程7はtoolkitを変更した開発時に使う。

## 1. 環境確認

数値比較だけなら、encoderを要求しない解析用preflightを使う。

```bash
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/check_environment.py" --analysis-only
```

Python 3.11以上、PyAV、Matplotlib、NumPy、Pillow、SciPyを検査する。`status=PASS`を確認する。このモードの `encoders` は空であり、動画出力の検証結果を示さない。

動画を制作する場合は、flagなしでencoderも含めて検査する。

```bash
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/check_environment.py"
```

`status`が`PASS`で、PyAV、Matplotlib、NumPy、Pillow、SciPy、`libx264`、AAC encoderがすべて`PASS`であることを確認する。
選択したモードの必須項目に不足がある場合は媒体を処理せず、依存物の準備についてユーザーへ説明する。

## 2. Clip manifestを固定する

[manifest-schema.md](manifest-schema.md#clip-manifest)のschema v1と区間数・再生順の不変条件に従う。

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

[manifest-schema.md](manifest-schema.md#render-manifest)のschema v2で作成する。
直前の解析runのfeature artifactsを実値hash付きで参照し、同schemaのanchor consistencyを確認する。`layout_review`はpreviewの明示承認後に追加する。

## 5. 動画とmappingを生成する

最初にユーザー提供のsynthetic layout referenceをviewし、integrity gateを通す。
この画像は証拠ではない。

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/verify_layout_reference.py"
```

`status=PASS`、asset SHA-256
`6d85aebbdefa38d886f8f78326e4ebb94b1aa66af797a9a0a4ac03c45311039a`、
1672×941 RGB PNGを確認する。
新規のimagegen案を作らない。

本番の前に、実録画を外部送信せずproduction renderer pathでpreviewを作る。
previewは実際のgroup名・色・指標rangeを使うが、波形とLog-Melは合成placeholderであり、その旨を画像内に表示する。

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$VOICE_PYTHON" "$VOICE_SKILL_DIR/scripts/render_layout_preview.py" \
  --manifest "/path/to/render_manifest.json" \
  --output "/path/to/output/layout_preview.png"
```

`layout_preview.review-sheet.png`と`layout_preview.png.provenance.json`も同時生成される。
[video-and-qa.md](video-and-qa.md#preview)に従い、review sheetを原寸で確認し、全checkがPASSしたpreviewの明示承認を得る。
承認後、[manifest-schema.md](manifest-schema.md#layout-review)の `layout_review` をrender manifestへ固定する。
基準asset、preview、review sheet、provenance、renderer source、basis hash、承認status、checkのいずれかが不一致ならrendererは本番前にFAILする。

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
QA JSONのhash欄にはlayout reference asset/manifest、承認preview、review sheet、preview provenanceも含まれる。
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
