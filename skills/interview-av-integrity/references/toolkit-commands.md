# Toolkit commands

`SKILL.md` からの相対位置で、toolkit rootは `scripts/toolkit` である。

- 初回準備: [runtime](#runtime)、[local word timestamps](#local-word-timestamps)、[case workspace](#case-workspace)
- 少数区間の予備診断: [analysis-protocol.md](analysis-protocol.md#予備診断)。手動測定には自動解析用のASR・face modelを要求しない
- 解析: [speaker/token manifest](#speakertoken-manifest)、[automated plosive analysis](#automated-plosive-analysis)
- 盲検注釈: [blinded review](#blinded-review)、[結合と視覚review](#結合directionclosure視覚blind-review)
- 動画制作: [evidence manifest/render/QA](#evidence-manifestrenderqa)

## runtime

Python 3.10–3.12を使う。解析案件のworkspace内に専用venvを作り、toolkitをeditable installする。

```bash
python3.12 -m venv CASE_DIR/.venv
CASE_DIR/.venv/bin/python -m pip install --no-build-isolation -e SKILL_DIR/scripts/toolkit
```

依存物取得にnetworkが必要なら、動画を送信しないことを説明して承認を得る。Face Landmarkerは公開repoに同梱しない。`scripts/toolkit/THIRD_PARTY_NOTICES.md`を確認し、ユーザーが明示承認した場合だけ次を実行する。

```bash
python SKILL_DIR/scripts/fetch_face_landmarker.py --accept-download
```

helperは公式URL以外へ接続せず、取得物のSHA-256が固定値と一致しない限り配置しない。

インストール後にruntimeとencoderを確認する。このcheckは案件媒体を読まず、合成フレーム1枚で実際のMediaPipe分離workerを起動する。`face_runtime.status` が `PASS` でなければ、同じ実行contextで本解析を開始しない。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/check_environment.py \
  --face-model SKILL_DIR/scripts/toolkit/models/face_landmarker.task \
  --output CASE_DIR/qa/environment.json
```

## local word timestamps

既存のword-level JSONがない場合は、ローカルASRを使う。Apple Siliconで `mlx_whisper` とローカルモデルがある例:

```bash
HF_HUB_OFFLINE=1 mlx_whisper /absolute/interview.mp4 \
  --model /absolute/local-whisper-model \
  --language ja --task transcribe \
  --word-timestamps True --output-format json \
  --output-dir CASE_DIR/transcript
```

モデルがローカルにない場合は、ダウンロード前に承認を得る。動画・音声を外部ASRへ送らない。別のローカルWhisper実装でも、segment内に `words[].word/start/end/probability` があるJSONへ整形すればよい。

完全分析では、さらに対象範囲の全wordへ `reading`、`kana`、`pronunciation` のいずれかを付与する。これはローカルの日本語形態素・読み解析器で行い、surface textだけを全件台帳とみなさない。

## case workspace

```bash
python SKILL_DIR/scripts/create_case_workspace.py \
  --video /absolute/interview.mp4 \
  --case-dir /absolute/case-dir
```

予備診断では、同じcommandの `--start START_S --end END_S` で対象範囲を記録できる。値は原本の秒時刻とし、離れた区間・対照の一覧は生成された `protocol/analysis_protocol.md` に固定する。
このhelperは `case.json` に原本fingerprintと範囲を保存するだけで、媒体を切り出したり後続analyzerの対象を絞ったりしない。media probeと実測は[予備診断手順](analysis-protocol.md#予備診断)に従う。

## speaker/token manifest

まず `assets/speaker-references.example.json` を案件用にコピーし、各speakerの表示名が安定する参照時刻とgroupを設定する。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/build_speaker_token_manifest.py \
  /absolute/interview.mp4 /absolute/whisper_words.json CASE_DIR/speakers \
  --references-json CASE_DIR/speaker_references.json \
  --require-complete-readings \
  --end INTERVIEW_END_S \
  --label-crop X0,Y0,X1,Y1
```

## automated plosive analysis

最初は2–3件のsmokeを行い、`processing_error`が全件でないことを確認する。
このrunnerはまず合成フレームでface runtimeをfail-fast preflightし、その後にH.264、1920×1080、24fps CFR、AAC 48kHzだけを受け付け、全video PTSを走査してCFRを確認する。別profileへ既定閾値を流用しない。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/analyze_plosive_sync.py \
  --video /absolute/interview.mp4 \
  --words-json /absolute/whisper_words.json \
  --intervals-json CASE_DIR/speakers/speaker_intervals.json \
  --speaker-token-manifest CASE_DIR/speakers/speaker_token_manifest.json \
  --face-model SKILL_DIR/scripts/toolkit/models/face_landmarker.task \
  --interview-end INTERVIEW_END_S \
  --output-dir CASE_DIR/automated \
  --max-events 3 --bootstrap-iterations 20 --permutation-iterations 20 --skip-plots
```

`--max-events 3` は先頭3件のeligible eventによる動作確認であり、対象・対照を選んだ予備診断や代表標本とはみなさない。
完全分析を依頼された場合だけ、smoke後に `--max-events`、反復数を20にする2つの指定、`--skip-plots` を外し、解析protocolに固定した反復数で対象範囲の全件を実行する。予備診断だけなら[analysis-protocol.md](analysis-protocol.md#予備診断)の選定区間・対照を測定して報告する。

POSIX shared memoryが許可されない環境では、同じコマンドの前に `VIDEO_INTEGRITY_FACE_TRANSPORT=file` を設定する。これはprivate temporary directoryのfile-backed mmapを使い、媒体を外部送信しない。

`DrishtiMetalHelper` と `Service is unavailable` を含む `face_runtime.status=FAIL` は、フレーム転送ではなくmacOSのnative graph serviceがmanaged sandbox内で起動できないことを示す。`VIDEO_INTEGRITY_FACE_TRANSPORT=file` では解決しない。そのrunを測定やsmoke PASSと扱わず、ユーザーの許可を得たローカル実行context（managed sandbox外）で、まずcheck、次にanalyzerを再実行する。ランドマーク方式を別の未校正detectorへ自動切替しない。

## blinded review

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/prepare_blinded_acoustic_review.py \
  --video /absolute/interview.mp4 \
  --events CASE_DIR/automated/events.json \
  --output-dir CASE_DIR/acoustic_blind
```

注釈CSVを固定後:

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/finalize_blinded_acoustic_review.py \
  --events CASE_DIR/automated/events.json \
  --blind-key CASE_DIR/acoustic_blind/blind_key.json \
  --annotations CASE_DIR/acoustic_blind/acoustic_annotations.csv \
  --output CASE_DIR/blinded_acoustic_annotations.json
```

## 結合、direction/closure、視覚blind review

音響盲検注釈と自動測定はevent IDだけで結合する。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/analyze_blinded_completeness.py \
  --annotations-json CASE_DIR/blinded_acoustic_annotations.json \
  --automated-events-json CASE_DIR/automated/events.json \
  --output-dir CASE_DIR/completeness

CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/analyze_plosive_direction_closure.py \
  --blinded-events-json CASE_DIR/blinded_acoustic_annotations.json \
  --automated-events-json CASE_DIR/automated/events.json \
  --output-dir CASE_DIR/direction_closure
```

次に、speaker/group/audio/機械分類を隠したnative-frame sheetとblind keyを作る。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/build_direction_closure_blind_sheets.py \
  --video /absolute/interview.mp4 \
  --events CASE_DIR/direction_closure/events.json \
  --output-dir CASE_DIR/visual_blind/sheets \
  --blind-key CASE_DIR/visual_blind/blind_key.json \
  --phoneme-class p \
  --roi X0 Y0 X1 Y1
```

各評価者は同一blind IDのCSVを独立に固定する。初回窓でno/after-only/曖昧な例は、必要に応じて後続+500ms窓を作る。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/build_direction_closure_followup_sheets.py \
  --video /absolute/interview.mp4 \
  --blind-key CASE_DIR/visual_blind/blind_key.json \
  --annotations CASE_DIR/visual_blind/reviewer1.csv \
  --output-dir CASE_DIR/visual_blind/followup \
  --roi X0 Y0 X1 Y1

CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/join_direction_closure_visual_audit.py \
  --events CASE_DIR/direction_closure/events.json \
  --blind-key CASE_DIR/visual_blind/blind_key.json \
  --annotations CASE_DIR/visual_blind/reviewer1.csv \
  --followup CASE_DIR/visual_blind/followup_annotations.csv \
  --output CASE_DIR/visual_blind/reviewer1_joined.csv
```

2名以上の評価者を反復 `--reviewer` で結合する。同数、過半数なし、曖昧票優勢は `ambiguous` とし、無理にyes/noへ丸めない。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/build_visual_contact_consensus.py \
  --reviewer CASE_DIR/visual_blind/reviewer1.csv \
  --reviewer CASE_DIR/visual_blind/reviewer2.csv \
  --metadata CASE_DIR/visual_blind/reviewer1_joined.csv \
  --output CASE_DIR/visual_blind/contact_consensus.csv
```

対象音素自体が疑わしい例は、映像を見ず音素分類sheetで再監査する。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/render_p_phoneme_classification_sheets.py \
  --video /absolute/interview.mp4 \
  --events CASE_DIR/direction_closure/events.json \
  --annotations CASE_DIR/blinded_acoustic_annotations.json \
  --output-dir CASE_DIR/phoneme_classification \
  --phoneme-class p
```

各scriptの出力countを固定値と思い込まずmanifestとID結合監査から確認する。

## evidence manifest/render/QA

`assets/event-manifest.example.json`を参考にcurated CSVを作り、次でmanifestを固定する。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/build_evidence_manifest.py \
  --video /absolute/interview.mp4 \
  --events-csv CASE_DIR/evidence_events.csv \
  --phone-label /p/ \
  --output CASE_DIR/evidence/event_manifest.json
```

イベントmanifest固定後、本番媒体を読まずにproduction-path previewとside-by-side review sheetを作る。
このscriptは承認済みsynthetic PNGとasset manifestのhash・寸法を先に検証し、本番 `render_evidence_frame` を合成プレースホルダで呼び出す。

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/render_layout_preview.py \
  --manifest CASE_DIR/evidence/event_manifest.json \
  --output CASE_DIR/evidence/layout_implementation_preview.png \
  --review-sheet CASE_DIR/evidence/layout_review_sheet.png \
  --report CASE_DIR/evidence/layout_preview_provenance.json
```

provenanceは `PENDING_USER_APPROVAL` の不変な承認前記録である。[layout-quality-gate.md](layout-quality-gate.md)に従い、review sheetで全checkを目視確認したpreviewだけをユーザーへ示し、そのpreviewの明示承認を得る。

承認後は[artifact-schemas.md](artifact-schemas.md#layout_review)の `layout_review` をevent manifestへ追加する。正確なquality check名とpath/hash形式は同schemaに従う。

レイアウトhard gateがPASSした後:

```bash
CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/render_closure_evidence_video.py \
  --manifest CASE_DIR/evidence/event_manifest.json \
  --video /absolute/interview.mp4 \
  --output CASE_DIR/evidence/closure_evidence_video.mp4

CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/verify_closure_evidence_video.py \
  --output-dir CASE_DIR/evidence

CASE_DIR/.venv/bin/python SKILL_DIR/scripts/toolkit/scripts/final_visual_qa_closure_video.py \
  --video CASE_DIR/evidence/closure_evidence_video.mp4 \
  --manifest CASE_DIR/evidence/closure_evidence_manifest.json \
  --frame-map CASE_DIR/evidence/closure_evidence_frame_map.csv \
  --output-dir CASE_DIR/evidence/final_visual_qa
```

全caseのcontact sheetを目視し、PASS報告を作る。スクリプト出力だけを目視PASSと呼ばない。
