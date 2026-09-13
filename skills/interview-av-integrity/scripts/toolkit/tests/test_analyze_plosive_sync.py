from __future__ import annotations

import importlib.util
import math
import os
from types import SimpleNamespace
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from video_integrity_analyzer.plosive_manifest import BilabialToken


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_plosive_sync.py"
SPEC = importlib.util.spec_from_file_location("analyze_plosive_sync", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PlosiveRunnerHelperTests(unittest.TestCase):
    def test_analysis_strata_are_predeclared_and_nonoverlapping(self) -> None:
        definitions = MODULE.analysis_definitions()
        self.assertEqual(
            [definition.key for definition in definitions],
            ["primary_p", "secondary_b", "combined_exploratory"],
        )
        self.assertEqual(definitions[0].phone_classes, ("p",))
        self.assertEqual(definitions[1].phone_classes, ("b",))
        self.assertEqual(definitions[2].phone_classes, ("p", "b"))

    def test_event_csv_row_exposes_release_and_burst_mouth_shape(self) -> None:
        record = {
            "selection": {
                "event_id": "pb-0001",
                "speaker": "candidate",
                "group": "candidate",
                "epoch_id": "e1",
                "phoneme_class": "p",
                "kana": "プ",
                "token_text": "プ",
                "token_start_s": 10.0,
                "token_end_s": 10.1,
                "anchor_s": 10.0,
                "asr_probability": 0.99,
                "eligible": True,
                "exclusion_reason": None,
            },
            "attempted": True,
            "status": "measurable",
            "measurable": True,
            "release_lag_ms": 91.0,
            "diagnostic_candidate_lag_ms": 95.0,
            "acoustic_auto_accepted": True,
            "acoustic_auto_acceptance_reasons": [],
            "exclusion_reasons": [],
            "measurement": {
                "combined_timing_uncertainty_ms": 21.0,
                "acoustic": {
                    "candidate_time_s": 10.012,
                    "release_time_s": 10.012,
                    "score": 5.0,
                    "confidence": "high",
                    "measurable": True,
                    "exclusion_reasons": [],
                },
                "visual": {
                    "contact_time_s": 9.9,
                    "candidate_time_s": 10.103,
                    "release_time_s": 10.103,
                    "closure_duration_ms": 203.0,
                    "timing_uncertainty_ms": 20.8,
                    "valid_frame_fraction": 1.0,
                    "quality_score": 0.9,
                    "measurable": True,
                    "exclusion_reasons": [],
                },
                "mouth_shape_at_acoustic_release": {
                    "category": "closed/contact",
                    "frame_time_s": 10.0,
                    "frame_time_error_ms": -12.0,
                    "pair_apertures": [0.01, 0.012, 0.014],
                    "median_aperture": 0.012,
                    "maximum_aperture": 0.014,
                    "aperture_spread": 0.004,
                    "mouth_width_px": 100.0,
                    "face_quality": 0.9,
                    "repeated_frame": False,
                    "valid": True,
                    "exclusion_reasons": [],
                },
            },
            "audio_window": {
                "start_s": 9.25,
                "end_s": 10.6,
                "coverage_fraction": 1.0,
                "warnings": [],
            },
            "lip_samples": [{"time_s": 10.0}, {"time_s": 10.041667}],
            "diagnostic_mouth_shape_at_top_acoustic_candidate": {
                "category": "closed/contact",
                "frame_time_s": 10.0,
                "frame_time_error_ms": -12.0,
                "pair_apertures": [0.01, 0.012, 0.014],
                "median_aperture": 0.012,
                "maximum_aperture": 0.014,
                "valid": True,
                "exclusion_reasons": [],
            },
            "processing_error": None,
        }
        row = MODULE.event_csv_row(record)
        self.assertEqual(row["release_lag_ms"], 91.0)
        self.assertEqual(row["diagnostic_candidate_lag_ms"], 95.0)
        self.assertEqual(row["mouth_shape_category_at_burst"], "closed/contact")
        self.assertEqual(row["top_candidate_mouth_shape_category"], "closed/contact")
        self.assertEqual(row["raw_lip_sample_count"], 2)

    def test_counts_keep_ineligible_and_unmeasurable_events_in_audit(self) -> None:
        records = [
            {
                "selection": {
                    "speaker": "candidate",
                    "group": "candidate",
                    "eligible": True,
                },
                "attempted": True,
                "measurable": True,
                "acoustic_auto_accepted": True,
                "exclusion_reasons": [],
                "measurement": {
                    "mouth_shape_at_acoustic_release": {"category": "transition"}
                },
            },
            {
                "selection": {
                    "speaker": "control_beta",
                    "group": "control",
                    "eligible": False,
                },
                "attempted": False,
                "measurable": False,
                "acoustic_auto_accepted": False,
                "exclusion_reasons": ["low_asr_probability"],
                "measurement": None,
            },
        ]
        counts = MODULE.audit_counts(records)
        self.assertEqual(counts["overall"]["selected"], 2)
        self.assertEqual(counts["overall"]["eligible"], 1)
        self.assertEqual(counts["overall"]["acoustic_auto_accepted"], 1)
        self.assertEqual(counts["overall"]["measurable"], 1)
        self.assertEqual(counts["exclusion_reasons"]["low_asr_probability"], 1)
        self.assertEqual(
            counts["mouth_shape_at_accepted_acoustic_release"]["transition"], 1
        )

    def test_measure_token_bounds_acoustic_attribution_with_asr_word_spans(self) -> None:
        import numpy as np

        from video_integrity_analyzer.plosive_sync import (
            AcousticReleaseEstimate,
            AudioWindow,
        )

        token = BilabialToken(
            event_id="pb-0001",
            speaker="candidate",
            group="candidate",
            epoch_id="e1",
            phoneme_class="p",
            kana="ポ",
            token_text="ポ",
            token_start_s=21.0,
            token_end_s=21.08,
            anchor_s=21.04,
            asr_probability=0.99,
            segment_id=1,
            segment_no_speech_probability=0.0,
            eligible=True,
            exclusion_reason=None,
            previous_word_window_s=(20.88, 21.0),
            next_word_window_s=(21.08, 21.14),
            word_occurrence_index=0,
            word_occurrence_anchors_s=(21.04,),
        )
        waveform = np.zeros(48_000, dtype=np.float64)
        audio = AudioWindow(
            start_s=20.29,
            sample_rate=48_000,
            waveform=waveform,
            coverage_mask=np.ones(len(waveform), dtype=bool),
            coverage_fraction=1.0,
        )
        acoustic = AcousticReleaseEstimate(
            anchor_time_s=21.04,
            phone_class="p",
            candidate_time_s=None,
            release_time_s=None,
            score=None,
            runner_up_margin=None,
            acceptance_mode="conservative",
            confidence="insufficient",
            measurable=False,
            time_resolution_ms=1.0,
            exclusion_reasons=("no_acoustic_candidate_within_target_word",),
            selected_candidate=None,
            candidates=(),
            target_window_s=(21.0, 21.08),
        )
        with patch.object(MODULE, "decode_audio_window", return_value=audio), patch.object(
            MODULE, "estimate_acoustic_release", return_value=acoustic
        ) as estimator, patch.object(
            MODULE, "extract_native_lip_samples", return_value=[]
        ):
            record = MODULE._measure_token(
                token,
                video=Path("synthetic.mp4"),
                face_model=Path("face.task"),
                roi=None,
                interview_end_s=60.0,
                acoustic_config=MODULE.AcousticReleaseConfig.conservative(),
                visual_config=MODULE.VisualReleaseConfig(),
            )
        kwargs = estimator.call_args.kwargs
        self.assertEqual(kwargs["target_window_s"], (21.0, 21.08))
        self.assertEqual(kwargs["previous_window_s"], (20.88, 21.0))
        self.assertEqual(kwargs["next_window_s"], (21.08, 21.14))
        self.assertEqual(kwargs["anchor_time_s"], 21.04)
        self.assertEqual(kwargs["target_occurrence_index"], 0)
        self.assertEqual(kwargs["target_occurrence_anchors_s"], (21.04,))
        self.assertFalse(record["measurable"])
        self.assertIsNone(record["diagnostic_mouth_shape_at_top_acoustic_candidate"])
        self.assertIn(
            "no_acoustic_candidate_within_target_word", record["exclusion_reasons"]
        )
        row = MODULE.event_csv_row(record)
        self.assertEqual(row["acoustic_target_window_s"], (21.0, 21.08))
        self.assertIsNone(row["acoustic_top_candidate_time_s"])
        self.assertIsNone(row["acoustic_candidate_attribution"])
        self.assertEqual(row["acoustic_target_occurrence_index"], 0)
        self.assertEqual(row["acoustic_target_occurrence_count"], 1)

    def test_roi_validation_rejects_out_of_bounds_values(self) -> None:
        self.assertEqual(MODULE._parse_roi([0.1, 0.2, 0.9, 0.8]), (0.1, 0.2, 0.9, 0.8))
        with self.assertRaises(ValueError):
            MODULE._parse_roi([0.2, 0.2, 1.2, 0.8])

    def test_nonfinite_missing_face_values_become_strict_json_null(self) -> None:
        converted = MODULE._json_ready(
            {"median_aperture": math.nan, "values": (1.0, math.inf)}
        )
        self.assertEqual(converted, {"median_aperture": None, "values": [1.0, None]})

    def test_empty_csv_tables_keep_placeholder_column(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "epochs.csv"
            MODULE._write_csv(target, [])
            self.assertEqual(target.read_bytes(), b'empty\r\n""\r\n')
            MODULE._write_csv(target, [{"a": [1, math.nan], "b": None}])
            self.assertEqual(target.read_bytes(), b'a,b\r\n"[1,null]",\r\n')
        self.assertEqual(MODULE._json_cell((1.0, math.inf)), "[1.0,null]")

    def test_helper_source_change_invalidates_checkpoint_with_same_size_and_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            package = project / "video_integrity_analyzer"
            package.mkdir()
            for name in (
                "artifact_io.py", "plosive_sync.py", "plosive_manifest.py", "plosive_stats.py"
            ):
                (package / name).write_bytes(
                    (MODULE.PROJECT / "video_integrity_analyzer" / name).read_bytes()
                )
            helper = package / "artifact_io.py"
            helper.write_bytes(helper.read_bytes() + b"\n# regression version a\n")
            source = project / "synthetic-input.bin"
            source.write_bytes(b"synthetic input")
            configuration_arguments = {
                "video": source,
                "words_json": source,
                "intervals_json": source,
                "speaker_token_manifest": source,
                "face_model": source,
                "roi": None,
                "interview_end_s": 1.0,
                "interview_end_source": "synthetic",
                "acoustic_config": MODULE.AcousticReleaseConfig.conservative(),
                "visual_config": MODULE.VisualReleaseConfig(),
                "token_inventory_scope": "synthetic",
                "frame_timing": {},
            }
            records = {"synthetic-event": {"attempted": True, "status": "measurable"}}
            checkpoint = project / "checkpoint.json"
            with patch.object(MODULE, "PROJECT", project):
                original = MODULE._measurement_configuration(**configuration_arguments)
                MODULE._write_json(
                    checkpoint,
                    {"configuration": original, "records_by_event_id": records},
                )
                self.assertEqual(
                    MODULE._load_checkpoint(checkpoint, original, resume=True), records
                )

                metadata = helper.stat()
                helper.write_bytes(
                    helper.read_bytes().replace(
                        b"# regression version a", b"# regression version b"
                    )
                )
                os.utime(helper, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
                self.assertEqual(helper.stat().st_size, metadata.st_size)
                self.assertEqual(helper.stat().st_mtime_ns, metadata.st_mtime_ns)
                updated = MODULE._measurement_configuration(**configuration_arguments)
                with self.assertRaisesRegex(
                    RuntimeError, "different inputs or measurement settings"
                ):
                    MODULE._load_checkpoint(checkpoint, updated, resume=True)

    def test_interview_end_defaults_to_media_duration(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "--video",
                "video.mp4",
                "--words-json",
                "words.json",
                "--intervals-json",
                "intervals.json",
                "--speaker-token-manifest",
                "speaker_tokens.json",
                "--face-model",
                "face.task",
                "--output-dir",
                "output",
            ]
        )
        self.assertIsNone(args.interview_end)
        self.assertEqual(
            MODULE._resolve_interview_end(None, media_duration_s=123.456),
            123.456,
        )

    def test_case_derived_media_profile_fails_closed(self) -> None:
        supported = SimpleNamespace(
            video_codec="h264",
            width=1920,
            height=1080,
            average_fps=24.0,
            audio_codec="aac",
            audio_sample_rate=48_000,
        )
        MODULE._validate_supported_media_profile(supported)
        unsupported = SimpleNamespace(**{**supported.__dict__, "average_fps": 30.0})
        with self.assertRaisesRegex(ValueError, "limited"):
            MODULE._validate_supported_media_profile(unsupported)
        self.assertEqual(
            MODULE._resolve_interview_end(100.0, media_duration_s=123.456),
            100.0,
        )
        with self.assertRaisesRegex(ValueError, "exceeds media duration"):
            MODULE._resolve_interview_end(124.0, media_duration_s=123.456)

    def test_token_level_speaker_guard_supersedes_interval_approximation(self) -> None:
        base = BilabialToken(
            event_id="pb-0001",
            speaker="candidate",
            group="candidate",
            epoch_id="e1",
            phoneme_class="p",
            kana="プ",
            token_text="プバ",
            token_start_s=10.0,
            token_end_s=10.4,
            anchor_s=10.1,
            asr_probability=0.99,
            segment_id=1,
            segment_no_speech_probability=0.0,
            eligible=False,
            exclusion_reason="near_active_speaker_transition",
        )
        second = replace(
            base,
            event_id="pb-0002",
            phoneme_class="b",
            kana="バ",
            anchor_s=10.3,
            eligible=True,
            exclusion_reason=None,
        )
        third = replace(
            base,
            event_id="pb-0003",
            token_text="プ",
            token_start_s=20.0,
            token_end_s=20.2,
            anchor_s=20.1,
            eligible=True,
            exclusion_reason=None,
        )
        manifest = {
            "tokens": [
                {
                    "token_id": "pb_0000",
                    "asr_word_start_s": 10.0,
                    "asr_word_end_s": 10.4,
                    "kana": "プ",
                    "occurrence_in_word": 0,
                    "speaker_id": "candidate",
                    "group": "candidate",
                    "speaker_stable_for_analysis": True,
                },
                {
                    "token_id": "pb_0001",
                    "asr_word_start_s": 10.0,
                    "asr_word_end_s": 10.4,
                    "kana": "バ",
                    "occurrence_in_word": 1,
                    "speaker_id": "candidate",
                    "group": "candidate",
                    "speaker_stable_for_analysis": False,
                },
                {
                    "token_id": "pb_0002",
                    "asr_word_start_s": 20.0,
                    "asr_word_end_s": 20.2,
                    "kana": "プ",
                    "occurrence_in_word": 0,
                    "speaker_id": "reviewer_zeta",
                    "group": "control",
                    "speaker_stable_for_analysis": True,
                },
            ]
        }
        joined, metadata, audit = MODULE.enforce_speaker_token_manifest(
            [base, second, third], manifest
        )
        self.assertTrue(joined[0].eligible)
        self.assertFalse(joined[1].eligible)
        self.assertEqual(
            joined[1].exclusion_reason, "speaker_token_manifest_unstable"
        )
        self.assertEqual(joined[2].speaker, "reviewer_zeta")
        self.assertEqual(joined[2].group, "control")
        self.assertTrue(metadata["pb-0003"]["speaker_label_mismatch"])
        self.assertEqual(audit["counts"]["speaker_label_mismatch"], 1)
        self.assertEqual(audit["counts"]["eligible_after_join"], 2)

    def test_manifest_reference_group_is_used_without_known_speaker_ids(self) -> None:
        token = BilabialToken(
            event_id="pb-0001",
            speaker="temporary_interval_label",
            group="control",
            epoch_id="e1",
            phoneme_class="p",
            kana="プ",
            token_text="プ",
            token_start_s=10.0,
            token_end_s=10.2,
            anchor_s=10.1,
            asr_probability=0.99,
            segment_id=1,
            segment_no_speech_probability=0.0,
            eligible=True,
            exclusion_reason=None,
        )
        manifest = {
            "method": {
                "reference_labels": {
                    "speaker_47": {
                        "display_name": "Reference Speaker",
                        "time_s": 5.0,
                        "group": "candidate",
                    }
                }
            },
            "tokens": [
                {
                    "token_id": "pb_0000",
                    "asr_word_start_s": 10.0,
                    "asr_word_end_s": 10.2,
                    "kana": "プ",
                    "occurrence_in_word": 0,
                    "speaker_id": "speaker_47",
                    "speaker_stable_for_analysis": True,
                }
            ],
        }
        joined, metadata, audit = MODULE.enforce_speaker_token_manifest(
            [token], manifest
        )
        self.assertTrue(joined[0].eligible)
        self.assertEqual(joined[0].speaker, "speaker_47")
        self.assertEqual(joined[0].group, "candidate")
        self.assertEqual(metadata["pb-0001"]["manifest_group"], "candidate")
        self.assertEqual(audit["counts"]["stable"], 1)

    def test_legacy_manifest_uses_interval_group_when_group_metadata_is_absent(self) -> None:
        token = BilabialToken(
            event_id="legacy-event",
            speaker="speaker_legacy",
            group="candidate",
            epoch_id="e1",
            phoneme_class="p",
            kana="プ",
            token_text="プ",
            token_start_s=10.0,
            token_end_s=10.2,
            anchor_s=10.1,
            asr_probability=0.99,
            segment_id=1,
            segment_no_speech_probability=0.0,
            eligible=True,
            exclusion_reason=None,
        )
        manifest = {
            "tokens": [
                {
                    "token_id": "legacy-token",
                    "asr_word_start_s": 10.0,
                    "asr_word_end_s": 10.2,
                    "kana": "プ",
                    "occurrence_in_word": 0,
                    "speaker_id": "speaker_legacy",
                    "speaker_stable_for_analysis": True,
                }
            ]
        }
        joined, metadata, _ = MODULE.enforce_speaker_token_manifest([token], manifest)
        self.assertTrue(joined[0].eligible)
        self.assertEqual(joined[0].group, "candidate")
        self.assertEqual(
            metadata["legacy-event"]["manifest_group_source"],
            "speaker_interval_fallback",
        )


if __name__ == "__main__":
    unittest.main()
