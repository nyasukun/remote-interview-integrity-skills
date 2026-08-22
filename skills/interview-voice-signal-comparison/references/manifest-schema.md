# Manifest schema

full pipelineでは、特徴抽出用のclip manifestと動画用render manifestを分ける。
どちらも絶対source pathを受け取れるが、skill自体へユーザー固有pathを埋め込まない。

## Clip manifest

```json
{
  "schema_version": 1,
  "analysis_type": "non_biometric_descriptive_acoustic_features",
  "source_provenance": [
    {
      "path": "/absolute/path/to/recording.mp4",
      "sha256": "<sha256>"
    }
  ],
  "groups": [
    {
      "group_id": "designated",
      "group_label": "ユーザー指定区間",
      "role": "designated_point",
      "display_color_hex": "#D55E00"
    },
    {
      "group_id": "reference-1",
      "group_label": "参照A",
      "role": "comparison_reference",
      "display_color_hex": "#0072B2"
    }
  ],
  "designated_group_id": "designated",
  "comparison_group_ids": ["reference-1"],
  "clips": [
    {
      "clip_id": "designated-01",
      "group_id": "designated",
      "source": "/absolute/path/to/recording.mp4",
      "start_s": 100.0,
      "end_s": 106.5,
      "language_label": "user supplied",
      "label_origin": "caller supplied; not audio inferred"
    },
    {
      "clip_id": "reference-1-01",
      "group_id": "reference-1",
      "source": "/absolute/path/to/recording.mp4",
      "start_s": 220.0,
      "end_s": 226.5,
      "language_label": "caller supplied",
      "label_origin": "display-name interval; not biometric identity"
    }
  ]
}
```

### Invariants

- `group_id`と`clip_id`は一意。
- `designated_group_id`は1 clipだけを持つ。
- `comparison_group_ids`は1〜3個で、各群は1 clip以上を持つ。
- `0 <= start_s < end_s`で、source音声範囲内。
- 全groupにrole、表示label、色を与える。
- language・speaker labelの由来を音声推定と誤認しない文言で保存する。

## Render manifest

```json
{
  "schema_version": 2,
  "title": "指定区間中心 音声信号比較",
  "clip_manifest": {
    "path": "/absolute/path/to/clips.json",
    "sha256": "<sha256>"
  },
  "feature_artifacts": {
    "acoustic_features": "/absolute/path/to/acoustic_features.json",
    "artifact_manifest": "/absolute/path/to/artifact_manifest.json"
  },
  "comparison": {
    "anchor_group": "designated",
    "display_anchor_group": "designated",
    "point_group": "designated",
    "reference_groups": ["reference-1"],
    "delta_definition": "group metric minus user-designated point metric"
  },
  "timeline": {
    "intro_seconds": 5.0,
    "gap_seconds": 0.75,
    "outro_seconds": 5.0,
    "edge_fade_ms": 10.0
  },
  "output": {
    "width": 1920,
    "height": 1080,
    "fps": 24,
    "audio_rate": 48000
  }
}
```

### Anchor consistency

`comparison.anchor_group`、`display_anchor_group`、`point_group`、summaryの`anchor_group`は同じ指定群を指す。
候補群や参照群のrangeを補助表示する場合も、`candidate_range_group`等の別名にし、anchorと呼ばない。
`relations_to_anchor`のように基準が曖昧なlegacy fieldを混在させない。

## Effective manifest

rendererは入力manifestを直接上書きせず、次を含むeffective manifestを別生成する。

- 解決済み絶対pathとhash
- sourceごとのSHA-256
- 実際にdecodeしたsample数・coverage
- clip output時刻
- feature configとsummary
- designated-centered signed delta
- layout modeとanchor ID
- audio gain・edge fade
- limitation text
- output MP4、frame map、audio mapのhash

PCM全配列はmanifestへ埋め込まない。
