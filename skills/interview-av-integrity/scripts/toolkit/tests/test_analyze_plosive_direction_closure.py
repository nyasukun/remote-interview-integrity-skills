from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_plosive_direction_closure.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_plosive_direction_closure", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample(
    time_s: float,
    state: str = "open",
    *,
    repeated: bool = False,
) -> dict[str, object]:
    if state == "contact":
        median, maximum = 0.010, 0.014
    elif state == "transition":
        median, maximum = 0.025, 0.029
    else:
        median, maximum = 0.050, 0.054
    return {
        "time_s": time_s,
        "median_aperture": median,
        "maximum_aperture": maximum,
        "aperture_spread": maximum - median,
        "mouth_width_px": 110.0,
        "face_quality": 0.9,
        "pair_apertures": [median - 0.001, median, maximum],
        "repeated_frame": repeated,
        "face_detected": True,
        "source_pts": round(time_s * 24),
        "source_time_base": "1/24",
    }


def sequence(audio_time: float, states: dict[int, str], repeated: set[int] | None = None):
    repeated = repeated or set()
    return [
        sample(
            audio_time + offset / 24.0,
            states.get(offset, "open"),
            repeated=offset in repeated,
        )
        for offset in range(-8, 9)
    ]


class DirectionClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MODULE.VisualReleaseConfig()

    def analyze(self, states: dict[int, str], repeated: set[int] | None = None):
        return MODULE.analyze_event_geometry(
            sequence(10.0, states, repeated),
            audio_time_s=10.0,
            fps=24.0,
            config=self.config,
        )

    def test_closure_classes_count_only_quality_valid_nonrepeated_frames(self) -> None:
        self.assertEqual(self.analyze({})["closure_category"], "no_visible_contact")
        self.assertEqual(
            self.analyze({-2: "contact"})["closure_category"], "weak_contact"
        )
        self.assertEqual(
            self.analyze({-3: "contact", -2: "contact"})["closure_category"],
            "strong_contact",
        )
        repeated = self.analyze(
            {-3: "contact", -2: "contact"}, repeated={-2}
        )
        self.assertEqual(repeated["closure_category"], "weak_contact")
        self.assertEqual(repeated["contact_nonrepeated_frame_count"], 1)

    def test_multiple_edges_are_retained_and_nearest_is_selected(self) -> None:
        result = self.analyze(
            {
                -5: "contact",
                -4: "open",
                2: "contact",
                3: "open",
            }
        )
        self.assertEqual(result["edge_candidate_count"], 2)
        self.assertTrue(result["ambiguous_multiple_edges"])
        self.assertEqual(result["direction"], "video_lags")
        self.assertAlmostEqual(result["selected_edge_lag_ms"], 104.1666667, places=4)
        self.assertLess(result["edge_candidates"][0]["edge_lag_ms"], -83.33)

    def test_contact_only_after_audio_is_flagged(self) -> None:
        result = self.analyze({2: "contact"})
        self.assertEqual(result["closure_category"], "weak_contact")
        self.assertTrue(result["post_audio_only_contact"])

    def test_contact_at_one_frame_after_audio_is_not_post_audio_only(self) -> None:
        result = self.analyze({1: "contact"})
        self.assertEqual(result["closure_category"], "weak_contact")
        self.assertFalse(result["post_audio_only_contact"])

    def test_bad_face_coverage_makes_event_unavailable_not_no_contact(self) -> None:
        bad = sequence(10.0, {})
        for row in bad:
            row["face_detected"] = False
        result = MODULE.analyze_event_geometry(
            bad, audio_time_s=10.0, fps=24.0, config=self.config
        )
        self.assertFalse(result["face_valid"])
        self.assertEqual(result["closure_category"], "unavailable")
        self.assertEqual(result["direction"], "unavailable")

    def test_isolated_open_frame_does_not_establish_absent_closure(self) -> None:
        result = MODULE.analyze_event_geometry(
            [sample(10.0)], audio_time_s=10.0, fps=24.0, config=self.config
        )
        self.assertFalse(result["face_valid"])
        self.assertEqual(result["closure_category"], "unavailable")
        self.assertEqual(result["closure_window_expected_native_frame_count"], 9)
        self.assertEqual(result["closure_window_native_frame_count"], 1)
        self.assertIn("missing_preclosure_window_coverage", result["face_exclusion_reasons"])

    def test_complete_pts_with_unusable_frames_does_not_establish_absent_closure(self) -> None:
        for unusable_offsets, failure in (({-3, -2, -1}, "missing_face"), ({-3, -2, -1}, "repeated"), ({-1}, "missing_face")):
            with self.subTest(offsets=unusable_offsets, failure=failure):
                rows = sequence(10.0, {})
                for row in rows:
                    if round((row["time_s"] - 10.0) * 24) in unusable_offsets:
                        if failure == "repeated":
                            row["repeated_frame"] = True
                        else:
                            row["face_detected"] = False
                result = MODULE.analyze_event_geometry(rows, audio_time_s=10.0, fps=24.0, config=self.config)
                self.assertEqual(result["closure_window_native_coverage_fraction"], 1.0)
                self.assertGreaterEqual(result["closure_window_usable_fraction"], 0.6)
                self.assertFalse(result["face_valid"])
                self.assertEqual(result["closure_category"], "unavailable")
                self.assertIn("incomplete_usable_frame_coverage_for_absence", result["face_exclusion_reasons"])

    def test_partial_usable_coverage_preserves_positive_contact_evidence(self) -> None:
        for states, expected in (({-5: "contact"}, "weak_contact"), ({-5: "contact", -4: "contact"}, "strong_contact")):
            for failure in ("missing_face", "repeated"):
                with self.subTest(states=states, failure=failure):
                    rows = sequence(10.0, states)
                    for row in rows:
                        if round((row["time_s"] - 10.0) * 24) in {-3, -2, -1}:
                            if failure == "repeated":
                                row["repeated_frame"] = True
                            else:
                                row["face_detected"] = False
                    result = MODULE.analyze_event_geometry(rows, audio_time_s=10.0, fps=24.0, config=self.config)
                    self.assertTrue(result["face_valid"])
                    self.assertEqual(result["closure_category"], expected)
                    self.assertAlmostEqual(result["closure_window_usable_fraction"], 6 / 9)
                    self.assertNotIn("incomplete_usable_frame_coverage_for_absence", result["face_exclusion_reasons"])

    def test_missing_preclosure_or_internal_native_frames_is_unavailable(self) -> None:
        full = sequence(10.0, {})
        for rows, reason in (
            ([row for row in full if row["time_s"] >= 10.0], "missing_preclosure_window_coverage"),
            ([row for row in full if abs(row["time_s"] - (10.0 - 3.0 / 24.0)) > 1e-6], "native_frame_gap_in_closure_window"),
        ):
            with self.subTest(reason=reason):
                result = MODULE.analyze_event_geometry(rows, audio_time_s=10.0, fps=24.0, config=self.config)
                self.assertEqual(result["closure_category"], "unavailable")
                self.assertIn(reason, result["closure_window_coverage_exclusion_reasons"])

    def test_complete_native_window_tolerates_pts_rounding(self) -> None:
        for states, expected in (({}, "no_visible_contact"), ({-2: "contact", -1: "contact"}, "strong_contact")):
            rows = sequence(10.0, states)
            for index, row in enumerate(rows):
                row["time_s"] = round(row["time_s"], 5) + (1e-5 if index % 2 else -1e-5)
            result = MODULE.analyze_event_geometry(rows, audio_time_s=10.0, fps=24.0, config=self.config)
            self.assertTrue(result["face_valid"])
            self.assertEqual(result["closure_category"], expected)
            self.assertEqual(result["closure_window_native_coverage_fraction"], 1.0)
            self.assertEqual(result["closure_window_coverage_exclusion_reasons"], [])

    def test_exact_permutation_has_no_monte_carlo_correction(self) -> None:
        result = MODULE.exact_one_sided_median_permutation(
            np.asarray([1.0]), np.asarray([0.0, 0.0])
        )
        self.assertEqual(result["total_labeling_count"], 3)
        self.assertEqual(result["tail_labeling_count"], 1)
        self.assertAlmostEqual(result["p_value"], 1.0 / 3.0)

    def test_exact_permutation_matches_naive_enumeration_with_ties(self) -> None:
        candidate = np.asarray([0.0, 1.0])
        control = np.asarray([0.0, 0.5, 1.0])
        combined = np.concatenate((candidate, control))
        observed = float(np.median(candidate) - np.median(control))
        statistics = []
        for indices in itertools.combinations(range(len(combined)), len(candidate)):
            mask = np.ones(len(combined), dtype=bool)
            mask[list(indices)] = False
            statistics.append(
                float(np.median(combined[list(indices)]) - np.median(combined[mask]))
            )
        expected = sum(value >= observed - 1e-12 for value in statistics) / len(
            statistics
        )
        result = MODULE.exact_one_sided_median_permutation(candidate, control)
        self.assertAlmostEqual(result["p_value"], expected)

    def test_direction_margin_boundaries_are_near(self) -> None:
        margin = 2000.0 / 24.0
        self.assertEqual(
            MODULE.classify_direction(-margin, margin_ms=margin), "near"
        )
        self.assertEqual(
            MODULE.classify_direction(margin, margin_ms=margin), "near"
        )
        self.assertEqual(
            MODULE.classify_direction(-margin - 0.001, margin_ms=margin),
            "video_leads",
        )
        self.assertEqual(
            MODULE.classify_direction(margin + 0.001, margin_ms=margin),
            "video_lags",
        )

    def test_run_uses_configurable_cutoff_and_writes_primary_and_secondary_outputs(self) -> None:
        def selection(event_id: str, phone: str, speaker: str, group: str):
            return {
                "event_id": event_id,
                "speaker": speaker,
                "group": group,
                "epoch_id": f"{speaker}-e1",
                "phoneme_class": phone,
                "token_start_s": 10.0,
            }

        definitions = [
            ("p-c", "p", "candidate", "candidate", 10.0),
            ("p-k", "p", "control_alpha", "control", 20.0),
            ("b-c2", "b", "control_b", "control", 30.0),
            ("late", "p", "candidate", "candidate", 138.0),
        ]
        blinded_rows = []
        automated_rows = []
        for event_id, phone, speaker, group, audio_time in definitions:
            selected = selection(event_id, phone, speaker, group)
            selected["token_start_s"] = audio_time
            blinded_rows.append(
                {
                    "runner_event_id": event_id,
                    "selection": selected,
                    "annotation_status": "measurable",
                    "audio_release_time_s": audio_time,
                    "annotation": {"confidence": "high"},
                }
            )
            automated_rows.append(
                {
                    "selection": selected,
                    "lip_samples": sequence(audio_time, {-2: "contact", -1: "contact"}),
                }
            )
        blinded = {"events": blinded_rows}
        automated = {
            "media": {"average_fps": 24.0},
            "configuration": {
                "protocol": {
                    "visual_release_config": asdict(self.config),
                }
            },
            "events": automated_rows,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blinded_path = root / "blinded.json"
            automated_path = root / "automated.json"
            output = root / "output"
            blinded_path.write_text(json.dumps(blinded), encoding="utf-8")
            automated_path.write_text(json.dumps(automated), encoding="utf-8")
            result = MODULE.run(
                argparse.Namespace(
                    blinded_events_json=blinded_path,
                    automated_events_json=automated_path,
                    output_dir=output,
                    bootstrap_iterations=100,
                    seed=7,
                    cutoff_s=137.5,
                    overwrite=False,
                )
            )
            self.assertEqual(
                result["join_and_eligibility_audit"][
                    "audio_measurable_in_scope_count"
                ],
                3,
            )
            self.assertEqual(result["configuration"]["cutoff_s_exclusive"], 137.5)
            self.assertAlmostEqual(
                result["configuration"]["closure_search_after_ms"],
                2000.0 / 24.0,
            )
            self.assertEqual(set(result["analyses"]), {"primary_p", "secondary_b"})
            self.assertTrue((output / "events.csv").is_file())
            rows = json.loads((output / "events.json").read_text(encoding="utf-8"))[
                "events"
            ]
            late = next(row for row in rows if row["runner_event_id"] == "late")
            self.assertFalse(late["analysis_eligible"])
            self.assertIn("token_at_or_after_cutoff", late["exclusion_reasons"])

    def test_loader_requires_canonical_events_key_and_empty_csv_is_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aliased = root / "aliased.json"
            aliased.write_text(json.dumps({"annotations": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "events list"):
                MODULE._load_json_object(aliased, "blinded events JSON")
            canonical = root / "canonical.json"
            canonical.write_text(json.dumps({"events": [], "media": {}}), encoding="utf-8")
            self.assertEqual(
                MODULE._load_json_object(canonical, "blinded events JSON")["events"], []
            )
            target = root / "epochs.csv"
            MODULE._write_csv(target, [])
            self.assertEqual(target.read_bytes(), b"\r\n")

    def test_build_records_accepts_acoustic_review_finalizer_schema(self) -> None:
        selected = {
            "event_id": "p-finalized",
            "speaker": "candidate",
            "group": "candidate",
            "epoch_id": "candidate-e1",
            "phoneme_class": "p",
            "token_start_s": 10.0,
        }
        blinded = {
            "events": [
                {
                    "runner_event_id": "p-finalized",
                    "status": "measurable",
                    "selected_release_time_s": 10.0,
                    "confidence": "high",
                    "speaker": "candidate",
                    "group": "candidate",
                    "epoch_id": "candidate-e1",
                    "phoneme_class": "p",
                }
            ]
        }
        automated = {
            "events": [
                {
                    "selection": selected,
                    "lip_samples": sequence(
                        10.0, {-2: "contact", -1: "contact"}
                    ),
                }
            ]
        }

        records, audit = MODULE.build_records(
            blinded,
            automated,
            fps=24.0,
            config=self.config,
        )

        self.assertEqual(audit["audio_measurable_in_scope_count"], 1)
        self.assertEqual(records[0]["annotation_status"], "measurable")
        self.assertEqual(records[0]["audio_release_time_s"], 10.0)
        self.assertEqual(records[0]["annotation_confidence"], "high")
        self.assertEqual(records[0]["selection"], selected)
        self.assertEqual(audit["legacy_unspecified_attribution_count"], 1)
        csv_row = MODULE.event_csv_row(records[0])
        self.assertEqual(csv_row["acoustic_attribution"], "legacy_unspecified")
        self.assertEqual(csv_row["annotation_confidence"], "high")
        self.assertIn("acoustic_realization", csv_row)

    def test_stale_finalized_automated_hash_is_rejected_before_join(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blind, automated = root / "blind.json", root / "automated.json"
            blind.write_text(json.dumps({"events": [], "metadata": {"input_sha256": {"automated_events": "stale"}}}))
            automated.write_text(json.dumps({"events": [], "media": {"average_fps": 24}}))
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                MODULE.run(argparse.Namespace(blinded_events_json=blind, automated_events_json=automated, output_dir=root / "output", overwrite=False))

    def test_geometry_protocol_controls_windows_and_direction_margin(self) -> None:
        protocol = MODULE.DirectionClosureProtocol(
            fps=30.0,
            cutoff_s=None,
            closure_before_s=0.10,
            closure_after_s=0.20,
            edge_before_s=0.15,
            edge_after_s=0.30,
            minimum_face_valid_fraction=0.50,
            maximum_nearest_frame_error_s=0.04,
            direction_margin_frames=3.0,
        )
        rows = [
            sample(10.0 + offset / 30.0, "contact" if offset == 1 else "open")
            for offset in range(-4, 10)
        ]
        result = MODULE.analyze_event_geometry(
            rows,
            audio_time_s=10.0,
            fps=30.0,
            config=self.config,
            protocol=protocol,
        )
        self.assertAlmostEqual(result["closure_window_start_s"], 9.9)
        self.assertAlmostEqual(result["closure_window_end_s"], 10.2)
        self.assertAlmostEqual(result["edge_search_start_s"], 9.85)
        self.assertAlmostEqual(result["edge_search_end_s"], 10.3)
        self.assertFalse(result["post_audio_only_contact"])

    def test_non_target_and_missing_audio_annotations_stay_excluded_in_audit(self) -> None:
        selected = {"event_id": "event-1", "phoneme_class": "p"}
        automated = {"events": [
            {"selection": selected, "lip_samples": sequence(10.0, {})},
            {"selection": {"event_id": "missing", "phoneme_class": "p"}},
        ]}
        for label in ("other", "uncertain", "b"):
            for nested in (False, True):
                row = {"runner_event_id": "event-1", "status": "measurable", "selected_release_time_s": 10.0}
                if nested:
                    row["annotation"] = {"acoustic_realization": label}
                else:
                    row["acoustic_realization"] = label
                with self.subTest(label=label, nested=nested):
                    records, audit = MODULE.build_records({"events": [row]}, automated, fps=24.0, config=self.config)
                    self.assertEqual(len(records), 2)
                    self.assertFalse(records[0]["analysis_eligible"])
                    self.assertIsNone(records[0]["geometry"])
                    self.assertIn("acoustic_realization_not_target", records[0]["exclusion_reasons"])
                    self.assertIn("missing_blinded_annotation", records[1]["exclusion_reasons"])
                    self.assertEqual(audit["automated_without_annotation_ids"], ["missing"])

    def test_direct_release_outside_review_window_is_not_measured(self) -> None:
        row = {"runner_event_id": "event-1", "status": "measurable", "selected_release_time_s": 10.0,
               "acoustic_realization": "p", "review_window": {"start_s": 8.0, "end_s": 9.0}}
        auto = {"events": [{"selection": {"event_id": "event-1", "phoneme_class": "p"}, "lip_samples": sequence(10.0, {})}]}
        records, _ = MODULE.build_records({"events": [row]}, auto, fps=24.0, config=self.config)
        self.assertFalse(records[0]["analysis_eligible"])
        self.assertIn("acoustic_release_outside_review_window", records[0]["exclusion_reasons"])


if __name__ == "__main__":
    unittest.main()
