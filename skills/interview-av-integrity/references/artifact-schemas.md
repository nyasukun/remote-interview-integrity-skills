# 成果物スキーマ

以下は論理スキーマであり、JSON/CSVの具体名はtoolkit出力に従ってよい。IDは全ファイルで一意かつ不変にする。

## source_probe.json

必須: `source_path`, `sha256`, `size_bytes`, `duration_s`, video/audio stream metadata, `time_base`, `start_pts`, `fps_mode`, `created_at`。

## analysis_protocol.md

対象範囲、話者割当、主副解析、閾値、除外、参照選定、統計単位、作成時刻、amendment履歴を含める。

## speaker_intervals.json

各行: `interval_id`, `epoch_id`, `start_s`, `end_s`, `speaker`, `group`, `assignment_signal`, `confidence`, `exclusion_reason`。

## token_manifest.json

各行: `event_id`, `word`, `reading`, `reading_source`, `kana`, `orthographic_class`, `intended_phone`, `anchor_s`, ASR word start/end/probability, `speaker`, `group`, `epoch_id`, `eligible`, `exclusion_reason`。
トップレベルに `token_inventory_scope=reading_complete` と読み欠落数を保存する。欠落がある場合は `literal_kana_or_supplied_reading_only` とし、全件台帳と呼ばない。

## acoustic_annotations.csv/json

各行: `blind_id`, `runner_event_id`, `status`, `selected_release_time_s`, `selected_release_relative_ms`, `selected_candidate_rank`, `acoustic_realization`, `confidence`, `reason`。

## visual_annotations.csv

各行: `blind_id`, `visible_contact`, `first_contact_timing`, `first_contact_offset_ms`, `contact_strength`, `confidence`, `note`。話者・群はunblindまで入れない。

## fused_events.csv/json

各行: 上記ID、speaker/group/epoch、音響時刻、視覚接触、可視release、signed lag、方向、品質、測定可能性、除外理由、注釈hash。

## event_manifest.json

トップレベル:

```json
{
  "schema_version": 1,
  "source_video": "/absolute/source.mp4",
  "title": "破裂音と口唇閉鎖の比較",
  "subtitle": "観測資料 — 原因判定ではありません",
  "phone_label": "/p/",
  "mouth_roi": [0.38, 0.50, 0.62, 0.78],
  "events": []
}
```

各event:

```json
{
  "event_id": "case-001",
  "source_release_s": 123.456,
  "classification": "closure_absent",
  "label": "対象語",
  "note": "可視フレーム内で明瞭な接触なし",
  "mouth_roi": [0.38, 0.50, 0.62, 0.78]
}
```

`phone_label` は動画上の表示用で、主解析なら `/p/`、副解析なら `/b/` などとする。
`classification` は `closure_absent`、`contact_reference`、`sync_reference` のいずれか。
参照が事前の同期基準を満たす場合だけ `sync_reference`とし、それ以外の可視接触例は `contact_reference`として「閉鎖あり参照」と表現する。

### layout_review

preview生成時には省略し、[layout-quality-gate.md](layout-quality-gate.md)の品質確認と明示承認後にだけ固定する。

承認後、event manifestのトップレベルへ次を追加する。pathはevent manifest基準の相対pathまたは絶対path、hashは実値を使う。

```json
{
  "layout_review": {
    "status": "APPROVED",
    "approval_basis": "user explicitly approved the displayed production-path preview",
    "preview": {
      "path": "layout_implementation_preview.png",
      "sha256": "<actual preview sha256>"
    },
    "preview_provenance": {
      "path": "layout_preview_provenance.json",
      "sha256": "<actual provenance sha256>"
    },
    "review_sheet": {
      "path": "layout_review_sheet.png",
      "sha256": "<actual review-sheet sha256>"
    },
    "quality_checks": {
      "reference_viewed_first": true,
      "visual_hierarchy_and_density_match": true,
      "case_finding_speed_and_legend_readable": true,
      "source_panel_remains_dominant": true,
      "mouth_roi_and_same_frame_inset_traceable": true,
      "mouth_crop_contains_lips_and_jaw": true,
      "closure_window_and_burst_emphasis_readable": true,
      "release_marker_and_playhead_readable": true,
      "japanese_text_readable_at_1920x1080": true,
      "limitations_and_uncertainty_readable": true,
      "production_renderer_preview": true,
      "side_by_side_review_completed": true
    }
  }
}
```

layout reference、manifest basis、renderer source、preview、provenance、review sheet、approval status、checkのいずれかが不一致・未完了なら、full-render CLIは案件媒体を読む前にFAILする。

## render artifacts

- `closure_evidence_video.mp4`
- `closure_evidence_manifest.json`: 実効窓、出力時刻、marker、crop、fps、カード長
- `closure_evidence_frame_map.csv`: output frameごとのsource PTS/index
- `closure_evidence_verification.json`, `QA_RESULTS.md`
- `FINAL_VISUAL_QA_summary.json`, contact sheets, audio RMS CSV
- 全ファイルSHA-256

絶対パスは内部監査用に保持できるが、外部共有版では必要に応じて相対化・秘匿する。
