from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BLIND = load_script("build_direction_closure_blind_sheets")
CONSENSUS = load_script("build_visual_contact_consensus")
JOIN = load_script("join_direction_closure_visual_audit")


def eligible_event(event_id: str) -> dict[str, object]:
    return {
        "runner_event_id": event_id,
        "selection": {"phoneme_class": "p"},
        "analysis_eligible": True,
        "geometry": {"face_valid": True},
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_consensus_metadata(path: Path, ids: tuple[str, ...]) -> None:
    write_csv(
        path,
        [
            {
                "blind_id": blind_id,
                "blind_order": str(index + 1),
                "runner_event_id": f"event-{index + 1}",
                "speaker": "candidate",
                "group": "candidate",
                "epoch_id": "epoch-1",
                "kana": "パ",
                "token_text": "token",
                "audio_release_time_s": str(index + 1.0),
                "machine_closure_category": "strong_contact",
            }
            for index, blind_id in enumerate(ids)
        ],
    )


class DirectionClosureReviewToolTests(unittest.TestCase):
    def test_blinding_accepts_arbitrary_nonempty_count_and_is_stable(self) -> None:
        payload = {
            "events": [eligible_event("event-c"), eligible_event("event-a")]
        }
        first = BLIND.blinded_events(payload)
        second = BLIND.blinded_events(payload)
        self.assertEqual(
            [row["runner_event_id"] for row in first],
            [row["runner_event_id"] for row in second],
        )
        self.assertEqual(len(first), 2)
        with self.assertRaises(RuntimeError):
            BLIND.blinded_events({"events": []})
        with self.assertRaises(RuntimeError):
            BLIND.blinded_events(payload, expected_count=3)

    def test_roi_and_dynamic_strip_rows_are_configurable(self) -> None:
        source = Image.new("RGB", (100, 80), "white")
        crop = BLIND.normalized_roi_crop(source, (0.1, 0.2, 0.9, 0.8))
        self.assertEqual(crop.size, (80, 48))
        frames = [(index / 30.0, source) for index in range(7)]
        rendered = BLIND.render_strip(
            "VC-001",
            0.1,
            frames,
            fps=30.0,
            roi=(0.1, 0.2, 0.9, 0.8),
            columns=3,
        )
        self.assertEqual(rendered.size, (930, 958))

    def test_review_contact_aliases_normalize_to_one_schema(self) -> None:
        for column in (
            "contact",
            "human_contact_full_window",
            "display_window_clear_contact",
            "clear_lip_contact",
        ):
            self.assertEqual(
                BLIND.normalized_review_value({column: " YES "}, "contact"),
                "yes",
            )
        with self.assertRaises(RuntimeError):
            BLIND.normalized_review_value(
                {"contact": "yes", "clear_lip_contact": "no"}, "contact"
            )
        self.assertIsNone(CONSENSUS.cohen_kappa(["yes"], ["yes"])[1])
        self.assertIsNone(
            CONSENSUS.fleiss_kappa([("yes", "yes", "yes")])[1]
        )
        self.assertEqual(
            CONSENSUS.strict_contact_majority(("ambiguous", "ambiguous", "yes")),
            "ambiguous",
        )
        self.assertEqual(
            CONSENSUS.strict_contact_majority(("yes", "yes", "ambiguous")),
            "yes",
        )

    def test_consensus_accepts_two_ids_and_mixed_legacy_headings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewer1 = root / "reviewer1.csv"
            reviewer2 = root / "reviewer2.csv"
            reviewer3 = root / "reviewer3.csv"
            metadata = root / "metadata.csv"
            output = root / "consensus.csv"
            ids = ("VC-001", "VC-002")
            write_csv(
                reviewer1,
                [
                    {
                        "blind_id": blind_id,
                        "contact": "yes" if index == 0 else "no",
                        "timing_relative_audio": "near",
                        "machine_window_category": "strong_contact",
                    }
                    for index, blind_id in enumerate(ids)
                ],
            )
            write_csv(
                reviewer2,
                [
                    {
                        "blind_id": blind_id,
                        "display_window_clear_contact": "yes" if index == 0 else "no",
                        "first_contact_timing": "near",
                        "machine_window_contact": "strong_contact",
                    }
                    for index, blind_id in enumerate(ids)
                ],
            )
            write_csv(
                reviewer3,
                [
                    {
                        "blind_id": blind_id,
                        "clear_lip_contact": "no",
                        "first_contact_timing": "near",
                        "contact_strength_250ms_to_83ms": "weak_contact",
                    }
                    for blind_id in ids
                ],
            )
            write_csv(
                metadata,
                [
                    {
                        "blind_id": blind_id,
                        "blind_order": str(index + 1),
                        "runner_event_id": f"event-{index + 1}",
                        "speaker": "candidate",
                        "group": "candidate",
                        "epoch_id": "epoch-1",
                        "kana": "パ",
                        "token_text": "token",
                        "audio_release_time_s": str(index + 1.0),
                        "machine_closure_category": "strong_contact",
                    }
                    for index, blind_id in enumerate(ids)
                ],
            )
            argv = [
                "build_visual_contact_consensus.py",
                "--reviewer1-blind",
                str(reviewer1),
                "--reviewer2-blind",
                str(reviewer2),
                "--reviewer3-blind",
                str(reviewer3),
                "--metadata",
                str(metadata),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                CONSENSUS.main()
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["contact_majority_2of3"], "yes")
            self.assertEqual(rows[1]["contact_majority_2of3"], "no")

    def test_consensus_repeated_reviewer_cli_supports_two_raters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = ("VC-001", "VC-002")
            reviewer_paths: list[Path] = []
            for reviewer_index, contacts in enumerate(
                (("contact", "yes"), ("no_contact", "yes")), start=1
            ):
                path = root / f"reviewer{reviewer_index}.csv"
                write_csv(
                    path,
                    [
                        {"blind_id": blind_id, "contact": contact}
                        for blind_id, contact in zip(ids, contacts, strict=True)
                    ],
                )
                reviewer_paths.append(path)
            metadata = root / "metadata.csv"
            output = root / "consensus.csv"
            write_consensus_metadata(metadata, ids)
            argv = ["build_visual_contact_consensus.py"]
            for path in reviewer_paths:
                argv.extend(("--reviewer", str(path)))
            argv.extend(("--metadata", str(metadata), "--output", str(output)))
            stdout = StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stdout(stdout):
                CONSENSUS.main()
            summary = json.loads(stdout.getvalue())
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary["rater_count"], 2)
            self.assertEqual(summary["contact_majority_threshold"], 2)
            self.assertEqual(len(summary["pairwise"]), 1)
            self.assertEqual(rows[0]["contact_majority"], "ambiguous")
            self.assertEqual(rows[1]["contact_majority"], "yes")
            self.assertEqual(rows[0]["contact_majority_2of3"], "")
            self.assertEqual(rows[0]["rater_count"], "2")
            self.assertNotIn("reviewer3_contact", rows[0])

    def test_consensus_repeated_reviewer_cli_supports_four_raters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = ("VC-001", "VC-002")
            ratings_by_reviewer = (
                ("yes", "yes"),
                ("contact", "yes"),
                ("yes", "no"),
                ("no_contact", "no"),
            )
            reviewer_paths: list[Path] = []
            for reviewer_index, contacts in enumerate(
                ratings_by_reviewer, start=1
            ):
                path = root / f"reviewer{reviewer_index}.csv"
                write_csv(
                    path,
                    [
                        {"blind_id": blind_id, "contact": contact}
                        for blind_id, contact in zip(ids, contacts, strict=True)
                    ],
                )
                reviewer_paths.append(path)
            metadata = root / "metadata.csv"
            output = root / "consensus.csv"
            write_consensus_metadata(metadata, ids)
            argv = ["build_visual_contact_consensus.py"]
            for path in reviewer_paths:
                argv.extend(("--reviewer", str(path)))
            argv.extend(("--metadata", str(metadata), "--output", str(output)))
            stdout = StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stdout(stdout):
                CONSENSUS.main()
            summary = json.loads(stdout.getvalue())
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary["rater_count"], 4)
            self.assertEqual(summary["contact_majority_threshold"], 3)
            self.assertEqual(len(summary["pairwise"]), 6)
            self.assertEqual(rows[0]["contact_majority"], "yes")
            self.assertEqual(rows[1]["contact_majority"], "ambiguous")
            self.assertEqual(rows[0]["reviewer4_contact"], "no")
            self.assertEqual(rows[0]["contact_majority_2of3"], "")

    def test_join_accepts_arbitrary_nonempty_audit_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_path = root / "events.json"
            key_path = root / "key.json"
            annotations_path = root / "annotations.csv"
            output = root / "joined.csv"
            ids = ("VC-001", "VC-002")
            events_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "runner_event_id": f"event-{index + 1}",
                                "selection": {
                                    "speaker": "candidate",
                                    "group": "candidate",
                                    "epoch_id": "epoch-1",
                                    "phoneme_class": "p",
                                    "kana": "パ",
                                    "token_text": "token",
                                },
                                "audio_release_time_s": index + 1.0,
                                "geometry": {
                                    "closure_category": "strong_contact",
                                    "contact_nonrepeated_frame_count": 2,
                                    "contact_frame_offsets_ms": [-20.0, 20.0],
                                    "post_audio_only_contact": False,
                                },
                            }
                            for index in range(2)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            key_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "blind_id": blind_id,
                                "runner_event_id": f"event-{index + 1}",
                            }
                            for index, blind_id in enumerate(ids)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                annotations_path,
                [
                    {
                        "blind_id": blind_id,
                        "contact": "yes",
                        "timing_relative_audio": "near",
                        "machine_window_category": "strong_contact",
                        "post_audio_only_contact": "false",
                        "first_clear_contact_offset_ms": "-20",
                        "confidence": "high",
                        "note": "",
                    }
                    for blind_id in ids
                ],
            )
            argv = [
                "join_direction_closure_visual_audit.py",
                "--events",
                str(events_path),
                "--blind-key",
                str(key_path),
                "--annotations",
                str(annotations_path),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                JOIN.main()
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["human_contact"], "yes")
            self.assertEqual(rows[0]["post_only_agreement"], "match")


if __name__ == "__main__":
    unittest.main()
