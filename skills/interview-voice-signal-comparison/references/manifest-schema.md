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
- 再生順は `clips` 配列順。既定では指定clipを先頭にし、比較clipを続ける。

選定後のmanifestを固定し、SHA-256を記録する。

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
    "acoustic_features": {
      "path": "/absolute/path/to/analysis_output/acoustic_features/acoustic_features.json",
      "sha256": "<actual sha256>"
    },
    "artifact_manifest": {
      "path": "/absolute/path/to/analysis_output/acoustic_features/artifact_manifest.json",
      "sha256": "<actual sha256>"
    }
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
  },
  "layout_review": {
    "status": "APPROVED",
    "approval_basis": "user explicitly approved the displayed production preview",
    "preview": {
      "path": "/absolute/path/to/layout_preview.png",
      "sha256": "<sha256>"
    },
    "preview_provenance": {
      "path": "/absolute/path/to/layout_preview.png.provenance.json",
      "sha256": "<sha256>"
    },
    "review_sheet": {
      "path": "/absolute/path/to/layout_preview.review-sheet.png",
      "sha256": "<sha256>"
    },
    "quality_checks": {
      "reference_viewed_first": true,
      "visual_hierarchy_matches": true,
      "designated_anchor_fixed_left": true,
      "comparison_panel_on_right": true,
      "waveform_and_logmel_readable": true,
      "shared_signed_delta_axes": true,
      "range_and_median_visible": true,
      "no_ranking_or_identity_claim": true,
      "dual_mono_treated_as_non_identifying": true,
      "limitation_strip_readable": true,
      "production_renderer_preview": true,
      "side_by_side_review_sheet_inspected": true
    }
  }
}
```

`feature_artifacts`は同じ解析runが生成した同一 `acoustic_features` ディレクトリの成果物を指す。
pathはrender manifest基準の相対pathまたは絶対path、hashは実値を使う。

### Layout review

`layout_review`はpreview生成時には省略する。
[video-and-qa.md](video-and-qa.md#preview)に従い、基準画像との比較・全quality check・ユーザーの明示承認が完了した後にだけ、上記の承認記録を追加する。
rendererは基準asset、承認済みpreview、review sheet、provenanceのhash、renderer source hash、render manifest basis、全check、承認statusをfail-closedで検証する。

### Anchor consistency

clip manifestの `designated_group_id`、`comparison.anchor_group`、`display_anchor_group`、`point_group`、summaryの`anchor_group`は同じ任意の指定群IDを指す。
`comparison.reference_groups`はclip manifestの `comparison_group_ids` と同じ順序、`delta_definition`は正確に `group metric minus user-designated point metric` とする。
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
- 承認済みlayout reference asset/manifestのpath・hash・寸法
- 承認済みproduction preview、side-by-side review sheet、preview provenance、renderer source hash、render manifest basis hash、quality checks、approval basis

PCM全配列はmanifestへ埋め込まない。
