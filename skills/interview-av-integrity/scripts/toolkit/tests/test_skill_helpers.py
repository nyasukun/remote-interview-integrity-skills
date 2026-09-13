from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SKILL_ROOT = Path(__file__).resolve().parents[3]
HELPERS = SKILL_ROOT / "scripts"


class SkillHelperTests(unittest.TestCase):
    def run_script(self, name: str, *arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(HELPERS / name), *(str(value) for value in arguments)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_create_case_workspace_is_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "source.mp4"
            video.write_bytes(b"unchanged-source")
            case = root / "case"
            first = self.run_script(
                "create_case_workspace.py", "--video", video, "--case-dir", case
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(video.read_bytes(), b"unchanged-source")
            config = json.loads((case / "case.json").read_text(encoding="utf-8"))
            self.assertEqual(config["source"]["path"], str(video.resolve()))
            second = self.run_script(
                "create_case_workspace.py", "--video", video, "--case-dir", case
            )
            self.assertNotEqual(second.returncode, 0)

    def test_model_fetch_requires_explicit_download_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script(
                "fetch_face_landmarker.py",
                "--output",
                Path(temporary) / "face_landmarker.task",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--accept-download", result.stderr)

    def test_evidence_manifest_sorts_novel_ids_and_audits_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "source.mp4"
            video.write_bytes(b"source")
            source_csv = root / "events.csv"
            with source_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "event_id",
                        "source_release_s",
                        "classification",
                        "label",
                        "note",
                        "include",
                        "exclusion_reason",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "event_id": "novel-reference",
                            "source_release_s": "20",
                            "classification": "contact_reference",
                            "label": "ref",
                            "note": "contact",
                            "include": "true",
                            "exclusion_reason": "",
                        },
                        {
                            "event_id": "novel-missing",
                            "source_release_s": "10",
                            "classification": "closure_absent",
                            "label": "missing",
                            "note": "no contact",
                            "include": "true",
                            "exclusion_reason": "",
                        },
                        {
                            "event_id": "novel-excluded",
                            "source_release_s": "15",
                            "classification": "closure_absent",
                            "label": "excluded",
                            "note": "ambiguous",
                            "include": "false",
                            "exclusion_reason": "indeterminate",
                        },
                    ]
                )
            output = root / "event_manifest.json"
            result = self.run_script(
                "build_evidence_manifest.py",
                "--video",
                video,
                "--events-csv",
                source_csv,
                "--output",
                output,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [event["event_id"] for event in payload["events"]],
                ["novel-missing", "novel-reference"],
            )
            audit = json.loads(
                (root / "event_manifest_selection_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["excluded_count"], 1)

    def test_finalize_acoustic_review_joins_only_after_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "events.json"
            events.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "selection": {
                                    "event_id": "novel-event",
                                    "speaker": "speaker-x",
                                    "group": "candidate",
                                    "epoch_id": "epoch-x",
                                    "phoneme_class": "p",
                                    "kana": "プ",
                                    "token_text": "プロジェクト",
                                    "anchor_s": 5.0,
                                },
                                "measurement": {"acoustic": {"candidates": [{"time_s": 5.01}]}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            key = root / "key.json"
            key.write_text(
                json.dumps(
                    {
                        "blinding": "speaker/group hidden",
                        "source_events_sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
                        "events": [
                            {
                                "blind_id": "AR-0001",
                                "runner_event_id": "novel-event",
                                "anchor_s": 5.0,
                                "review_window": {"start_s": 4.55, "end_s": 5.35, "covered_intervals": [{"start_s": 4.55, "end_s": 5.35}]},
                                "candidates": [
                                    {"rank": 1, "time_s": 5.01, "relative_ms": 10.0}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            annotations = root / "annotations.csv"
            with annotations.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "blind_id",
                        "status",
                        "selected_candidate_rank",
                        "selected_release_relative_ms",
                        "acoustic_realization",
                        "confidence",
                        "reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "blind_id": "AR-0001",
                        "status": "measurable",
                        "selected_candidate_rank": "1",
                        "selected_release_relative_ms": "10",
                        "acoustic_realization": "p",
                        "confidence": "high",
                        "reason": "clear release",
                    }
                )
            output = root / "fixed.json"
            result = self.run_script(
                "finalize_blinded_acoustic_review.py",
                "--events",
                events,
                "--blind-key",
                key,
                "--annotations",
                annotations,
                "--output",
                output,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(output.read_text(encoding="utf-8"))["events"][0]
            self.assertEqual(row["runner_event_id"], "novel-event")
            self.assertAlmostEqual(row["selected_release_time_s"], 5.01)
            self.assertEqual(row["speaker"], "speaker-x")

    def test_spectrogram_helper_is_finite_and_nonempty(self) -> None:
        path = HELPERS / "prepare_blinded_acoustic_review.py"
        spec = importlib.util.spec_from_file_location("prepare_blind_audio", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        waveform = np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
        matrix = module.spectrogram(waveform, 48000)
        self.assertGreater(matrix.shape[0], 10)
        self.assertGreater(matrix.shape[1], 10)
        self.assertTrue(np.isfinite(matrix).all())


if __name__ == "__main__":
    unittest.main()
