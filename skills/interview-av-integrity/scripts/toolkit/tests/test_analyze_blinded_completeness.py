from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_blinded_completeness.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_blinded_completeness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def lip_sample(time_s: float, category: str = "closed/contact") -> dict[str, object]:
    if category == "closed/contact":
        median, maximum = 0.010, 0.013
    elif category == "open":
        median, maximum = 0.050, 0.054
    else:
        median, maximum = 0.025, 0.029
    return {
        "time_s": time_s,
        "median_aperture": median,
        "maximum_aperture": maximum,
        "aperture_spread": maximum - median,
        "mouth_width_px": 110.0,
        "face_quality": 0.9,
        "pair_apertures": [median - 0.001, median, maximum],
        "repeated_frame": False,
        "face_detected": True,
        "track_id": 0,
        "source_pts": int(time_s * 24),
        "source_time_base": "1/24",
    }


def automated_event(
    event_id: str,
    *,
    speaker: str,
    group: str,
    epoch_id: str,
    phoneme_class: str,
    audio_time_s: float,
    visual_time_s: float | None,
    shape: str = "closed/contact",
) -> dict[str, object]:
    return {
        "selection": {
            "event_id": event_id,
            "speaker": speaker,
            "group": group,
            "epoch_id": epoch_id,
            "phoneme_class": phoneme_class,
            "anchor_s": audio_time_s,
        },
        "measurement": {
            "visual": {
                "release_time_s": visual_time_s,
                "measurable": visual_time_s is not None,
            }
        },
        "lip_samples": [
            lip_sample(audio_time_s - 1.0 / 24.0, "open"),
            lip_sample(audio_time_s, shape),
            lip_sample(audio_time_s + 1.0 / 24.0, "open"),
        ],
    }


def annotation(
    event_id: str,
    *,
    release_time_s: float | None,
    phoneme_class: str,
    status: str = "measurable",
) -> dict[str, object]:
    return {
        "runner_event_id": event_id,
        "status": status,
        "selected_release_time_s": release_time_s,
        "confidence": "high",
        "phoneme_class": phoneme_class,
    }


def burst_record(
    event_id: str,
    *,
    speaker: str,
    group: str,
    epoch_id: str,
    category: str,
    mar: float,
    lag_ms: float | None,
    phoneme_class: str = "p",
    valid: bool = True,
    annotation_status: str = "measurable",
) -> dict[str, object]:
    return {
        "runner_event_id": event_id,
        "selection": {
            "speaker": speaker,
            "group": group,
            "epoch_id": epoch_id,
            "phoneme_class": phoneme_class,
        },
        "annotation_present": True,
        "annotation_status": annotation_status,
        "lag_ms": lag_ms,
        "mouth_shape_at_audio_release": {
            "valid": valid,
            "category": category if valid else "unavailable",
            "median_aperture": mar if valid else None,
        },
    }


class BlindedCompletenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.visual_config = MODULE.VisualReleaseConfig()
        self.automated_events = [
            automated_event(
                "p-candidate",
                speaker="candidate",
                group="candidate",
                epoch_id="candidate-e1",
                phoneme_class="p",
                audio_time_s=10.0,
                visual_time_s=10.100,
                shape="closed/contact",
            ),
            automated_event(
                "p-control-1",
                speaker="control_alpha",
                group="control",
                epoch_id="control-1-e1",
                phoneme_class="p",
                audio_time_s=20.0,
                visual_time_s=20.040,
                shape="open",
            ),
            automated_event(
                "b-control-2",
                speaker="control_b",
                group="control",
                epoch_id="control-2-e1",
                phoneme_class="b",
                audio_time_s=30.0,
                visual_time_s=30.060,
                shape="transition",
            ),
            automated_event(
                "b-missing",
                speaker="candidate",
                group="candidate",
                epoch_id="candidate-e2",
                phoneme_class="b",
                audio_time_s=40.0,
                visual_time_s=None,
            ),
        ]
        self.automated = {
            "media": {"average_fps": 24.0},
            "configuration": {
                "protocol": {
                    "visual_release_config": {
                        key: value
                        for key, value in MODULE.asdict(self.visual_config).items()
                    }
                }
            },
            "events": self.automated_events,
        }
        self.annotations = {
            "events": [
                annotation("p-candidate", release_time_s=10.0, phoneme_class="p"),
                annotation("p-control-1", release_time_s=20.0, phoneme_class="p"),
                annotation("b-control-2", release_time_s=30.0, phoneme_class="b"),
            ]
        }

    def test_strict_join_combines_fixed_releases_and_raw_mouth_shape(self) -> None:
        records, audit = MODULE.join_blinded_annotations(
            self.annotations,
            self.automated,
            visual_config=self.visual_config,
        )
        candidate = next(
            record for record in records if record["runner_event_id"] == "p-candidate"
        )
        self.assertTrue(candidate["measurable"])
        self.assertAlmostEqual(candidate["lag_ms"], 100.0)
        self.assertEqual(
            candidate["mouth_shape_at_audio_release"]["category"],
            "closed/contact",
        )
        self.assertTrue(candidate["mouth_shape_at_audio_release"]["valid"])
        self.assertEqual(audit["matched_annotation_count"], 3)
        self.assertEqual(audit["automated_without_annotation_count"], 1)
        self.assertEqual(audit["automated_without_annotation_ids"], ["b-missing"])

    def test_missing_fixed_visual_release_is_counted(self) -> None:
        annotations = {
            "events": [
                *self.annotations["events"],
                annotation("b-missing", release_time_s=40.0, phoneme_class="b"),
            ]
        }
        records, audit = MODULE.join_blinded_annotations(
            annotations, self.automated, visual_config=self.visual_config
        )
        missing = next(
            record for record in records if record["runner_event_id"] == "b-missing"
        )
        self.assertFalse(missing["measurable"])
        self.assertIn("missing_fixed_visual_release", missing["exclusion_reasons"])
        self.assertEqual(audit["audio_measurable_missing_visual_count"], 1)

    def test_duplicate_unknown_and_class_mismatch_are_errors(self) -> None:
        duplicated = {"events": [self.annotations["events"][0]] * 2}
        with self.assertRaisesRegex(ValueError, "duplicate annotation"):
            MODULE.join_blinded_annotations(
                duplicated, self.automated, visual_config=self.visual_config
            )
        unknown = {
            "events": [annotation("unknown", release_time_s=1.0, phoneme_class="p")]
        }
        with self.assertRaisesRegex(ValueError, "unknown runner_event_id"):
            MODULE.join_blinded_annotations(
                unknown, self.automated, visual_config=self.visual_config
            )
        mismatch = {
            "events": [
                annotation("p-candidate", release_time_s=10.0, phoneme_class="b")
            ]
        }
        with self.assertRaisesRegex(ValueError, "phoneme_class mismatch"):
            MODULE.join_blinded_annotations(
                mismatch, self.automated, visual_config=self.visual_config
            )

    def test_status_and_release_time_must_be_consistent(self) -> None:
        missing_time = {
            "events": [
                annotation("p-candidate", release_time_s=None, phoneme_class="p")
            ]
        }
        with self.assertRaisesRegex(ValueError, "requires a finite"):
            MODULE.join_blinded_annotations(
                missing_time, self.automated, visual_config=self.visual_config
            )
        unexpected_time = {
            "events": [
                annotation(
                    "p-candidate",
                    release_time_s=10.0,
                    phoneme_class="p",
                    status="unmeasurable",
                )
            ]
        }
        with self.assertRaisesRegex(ValueError, "must not contain"):
            MODULE.join_blinded_annotations(
                unexpected_time, self.automated, visual_config=self.visual_config
            )

    def test_posthoc_descriptives_keep_control_speakers_separate(self) -> None:
        records, _ = MODULE.join_blinded_annotations(
            self.annotations, self.automated, visual_config=self.visual_config
        )
        posthoc, epoch_rows, speaker_rows, difference_rows = (
            MODULE.build_posthoc_descriptives(records)
        )
        combined_speakers = {
            row["speaker"]: row
            for row in speaker_rows
            if row["analysis"] == "combined_exploratory"
        }
        self.assertIn("control_b", combined_speakers)
        self.assertEqual(combined_speakers["control_b"]["event_count"], 1)
        self.assertAlmostEqual(
            combined_speakers["control_b"]["absolute_median_lag_ms"], 60.0
        )
        combined_difference = next(
            row
            for row in difference_rows
            if row["analysis"] == "combined_exploratory"
        )
        self.assertAlmostEqual(combined_difference["candidate_minus_control_ms"], 50.0)
        self.assertEqual(posthoc["status"], "post_hoc_descriptive_only")
        self.assertEqual(len([row for row in epoch_rows if row["analysis"] == "combined_exploratory"]), 3)

    def test_exact_median_permutation_matches_naive_label_enumeration(self) -> None:
        candidate = MODULE.np.asarray([0.5, 1.0], dtype=MODULE.np.float64)
        control = MODULE.np.asarray([0.0, 0.5, 0.75], dtype=MODULE.np.float64)
        exact = MODULE._exact_one_sided_median_permutation(candidate, control)
        combined = MODULE.np.sort(MODULE.np.concatenate((candidate, control)))
        observed = float(MODULE.np.median(candidate) - MODULE.np.median(control))
        tail = 0
        total = 0
        all_indices = set(range(combined.size))
        for selected in itertools.combinations(range(combined.size), candidate.size):
            remaining = sorted(all_indices - set(selected))
            statistic = float(
                MODULE.np.median(combined[list(selected)])
                - MODULE.np.median(combined[remaining])
            )
            tail += int(statistic >= observed - 1e-12)
            total += 1
        self.assertEqual(exact["total_labeling_count"], total)
        self.assertEqual(exact["tail_labeling_count"], tail)
        self.assertAlmostEqual(exact["p_value"], tail / total)

    def test_burst_shape_analysis_uses_epochs_and_keeps_speakers_separate(self) -> None:
        records = [
            burst_record(
                "c1-open",
                speaker="candidate",
                group="candidate",
                epoch_id="c-e1",
                category="open",
                mar=0.050,
                lag_ms=100.0,
            ),
            burst_record(
                "c1-transition",
                speaker="candidate",
                group="candidate",
                epoch_id="c-e1",
                category="transition",
                mar=0.025,
                lag_ms=100.0,
            ),
            burst_record(
                "c2-open",
                speaker="candidate",
                group="candidate",
                epoch_id="c-e2",
                category="open",
                mar=0.050,
                lag_ms=10.0,
            ),
            burst_record(
                "n-closed",
                speaker="control_alpha",
                group="control",
                epoch_id="n-e1",
                category="closed/contact",
                mar=0.010,
                lag_ms=100.0,
            ),
            burst_record(
                "j-transition",
                speaker="control_b",
                group="control",
                epoch_id="j-e1",
                category="transition",
                mar=0.025,
                lag_ms=-100.0,
            ),
            burst_record(
                "invalid",
                speaker="control_b",
                group="control",
                epoch_id="j-e1",
                category="open",
                mar=0.050,
                lag_ms=-100.0,
                valid=False,
            ),
        ]
        result, epoch_rows, speaker_rows, comparison_rows = (
            MODULE.build_burst_shape_analysis(
                records, fps=24.0, bootstrap_iterations=100, seed=42
            )
        )
        primary = result["analyses"]["primary_p"]
        self.assertEqual(primary["definition"]["inferential_role"], "primary")
        self.assertEqual(primary["eligibility"]["audio_measurable_event_count"], 6)
        self.assertEqual(primary["eligibility"]["eligible_event_count"], 5)
        speakers = {
            row["speaker"]: row
            for row in speaker_rows
            if row["analysis"] == "primary_p"
        }
        self.assertEqual(speakers["candidate"]["event_count"], 3)
        self.assertEqual(speakers["candidate"]["open_count"], 2)
        self.assertEqual(speakers["candidate"]["transition_count"], 1)
        self.assertAlmostEqual(speakers["candidate"]["open_rate"], 2.0 / 3.0)
        self.assertEqual(speakers["control_b"]["event_count"], 1)
        candidate_first_epoch = next(
            row
            for row in epoch_rows
            if row["analysis"] == "primary_p" and row["epoch_id"] == "c-e1"
        )
        self.assertEqual(candidate_first_epoch["event_count"], 2)
        self.assertAlmostEqual(candidate_first_epoch["open_rate"], 0.5)
        self.assertAlmostEqual(candidate_first_epoch["median_mar"], 0.0375)
        open_comparison = next(
            row
            for row in comparison_rows
            if row["analysis"] == "primary_p" and row["metric"] == "open_rate"
        )
        self.assertEqual(open_comparison["candidate_epoch_count"], 2)
        self.assertEqual(open_comparison["control_epoch_count"], 2)
        self.assertAlmostEqual(
            open_comparison["median_difference_candidate_minus_control"], 0.75
        )
        self.assertAlmostEqual(open_comparison["probability_candidate_greater"], 1.0)
        self.assertAlmostEqual(open_comparison["cliffs_delta"], 1.0)
        self.assertEqual(open_comparison["exact_permutation_total_labeling_count"], 6)
        self.assertIsNotNone(open_comparison["cluster_bootstrap_ci95"])
        qa = primary["visual_release_direction_consistency_qa"]["counts"]
        self.assertEqual(qa["burst_shape_eligible_event_count"], 5)
        self.assertEqual(qa["within_one_frame_count"], 1)
        self.assertEqual(qa["positive_consistent_count"], 2)
        self.assertEqual(qa["positive_inconsistent_count"], 1)
        self.assertEqual(qa["negative_inconsistent_count"], 1)
        self.assertEqual(qa["consistent_count"], 3)
        self.assertEqual(qa["inconsistent_count"], 2)
        self.assertFalse(result["token_level_fisher_test"]["computed"])

    def test_burst_shape_includes_valid_audio_event_without_visual_release(self) -> None:
        records = [
            burst_record(
                "audio-only",
                speaker="candidate",
                group="candidate",
                epoch_id="c-e1",
                category="open",
                mar=0.050,
                lag_ms=None,
            )
        ]
        result, _, speaker_rows, _ = MODULE.build_burst_shape_analysis(
            records, fps=24.0, bootstrap_iterations=10, seed=7
        )
        candidate = next(
            row
            for row in speaker_rows
            if row["analysis"] == "primary_p" and row["speaker"] == "candidate"
        )
        self.assertEqual(candidate["event_count"], 1)
        qa = result["analyses"]["primary_p"][
            "visual_release_direction_consistency_qa"
        ]["counts"]
        self.assertEqual(qa["lag_unavailable_count"], 1)

    def test_run_writes_json_and_csv_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            annotation_path = root / "annotations.json"
            automated_path = root / "automated.json"
            output_dir = root / "output"
            annotation_path.write_text(
                json.dumps(self.annotations), encoding="utf-8"
            )
            automated_path.write_text(json.dumps(self.automated), encoding="utf-8")
            args = argparse.Namespace(
                annotations_json=annotation_path,
                automated_events_json=automated_path,
                output_dir=output_dir,
                bootstrap_iterations=20,
                permutation_iterations=20,
                seed=123,
                overwrite=False,
            )
            analysis = MODULE.run(args)
            expected = {
                "analysis.json",
                "events.json",
                "events.csv",
                "epoch_estimates.csv",
                "comparisons.csv",
                "posthoc_epoch_absolute.csv",
                "posthoc_speaker_descriptives.csv",
                "posthoc_candidate_control_absolute_difference.csv",
                "burst_shape_analysis.json",
                "burst_shape_epochs.csv",
                "burst_shape_speakers.csv",
                "burst_shape_comparisons.csv",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            loaded = json.loads((output_dir / "analysis.json").read_text())
            self.assertEqual(loaded["join_audit"]["status"], "passed")
            self.assertIn("signed_lag_analyses", analysis)
            self.assertIn("post_hoc_descriptive", analysis)
            self.assertIn("burst_shape_analysis", analysis)
            burst_shape = json.loads(
                (output_dir / "burst_shape_analysis.json").read_text()
            )
            self.assertEqual(
                burst_shape["analyses"]["primary_p"]["definition"][
                    "inferential_role"
                ],
                "primary",
            )

    def test_loader_accepts_annotations_alias(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotations.json"
            path.write_text(
                json.dumps({"annotations": self.annotations["events"]}),
                encoding="utf-8",
            )
            loaded = MODULE._load_json_object(path, "annotations JSON")
            self.assertEqual(loaded["events"], self.annotations["events"])


if __name__ == "__main__":
    unittest.main()
