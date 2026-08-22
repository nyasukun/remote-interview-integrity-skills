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


if __name__ == "__main__":
    unittest.main()
