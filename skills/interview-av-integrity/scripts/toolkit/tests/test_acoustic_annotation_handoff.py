from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from video_integrity_analyzer.acoustic_annotations import acoustic_annotation_guard, validate_automated_source

HELPERS = Path(__file__).resolve().parents[3] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HELPERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZE = load("finalize_blinded_acoustic_review")
PREPARE = load("prepare_blinded_acoustic_review")


class AcousticHandoffTests(unittest.TestCase):
    def setUp(self):
        self.source = {"events": [{
            "selection": {"event_id": "pb-1", "anchor_s": 1.0, "phoneme_class": "p", "eligible": True},
            "measurement": {"acoustic": {"candidates": [{"time_s": 1.02}]}},
        }]}
        self.key_row = {
            "blind_id": "AR-1", "runner_event_id": "pb-1", "anchor_s": 1.0,
            "review_window": {"start_s": 0.55, "end_s": 1.35, "covered_intervals": [{"start_s": 0.55, "end_s": 1.35}]},
            "candidates": [{"time_s": 1.02, "relative_ms": 20.0}],
        }
        self.annotation = {
            "blind_id": "AR-1", "status": "measurable", "acoustic_realization": "p",
            "confidence": "high", "reason": "audible target release",
            "selected_candidate_rank": "1", "selected_release_relative_ms": "",
        }

    def finalize(self, *, key_row=None, annotation=None, source_hash=None, duplicate=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events, key, annotations, output = [root / name for name in ("events.json", "key.json", "annotations.csv", "output.json")]
            events.write_text(json.dumps(self.source))
            row = copy.deepcopy(self.key_row if key_row is None else key_row)
            rows = [row, {**row, "blind_id": "AR-2"}] if duplicate else [row]
            key.write_text(json.dumps({
                "source_events_sha256": source_hash or hashlib.sha256(events.read_bytes()).hexdigest(),
                "events": rows,
            }))
            with annotations.open("w", newline="") as handle:
                record = self.annotation if annotation is None else annotation
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerow(record)
            argv = ["finalize", "--events", str(events), "--blind-key", str(key), "--annotations", str(annotations), "--output", str(output)]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                FINALIZE.main()
            return json.loads(output.read_text())

    def test_fixed_release_preserves_review_bounds(self):
        result = self.finalize()
        self.assertEqual(result["events"][0]["selected_release_time_s"], 1.02)
        self.assertEqual(result["events"][0]["review_window"], self.key_row["review_window"])

    def test_automated_provenance_verified_or_explicitly_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "automated.json"
            path.write_text(json.dumps(self.source))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            verified = validate_automated_source({"metadata": {"input_sha256": {"automated_events": digest}}}, path)
            self.assertEqual(verified["status"], "verified")
            self.assertEqual(validate_automated_source({}, path)["status"], "legacy_missing")
            for wrong in ("stale", None, ""):
                with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                    validate_automated_source({"metadata": {"input_sha256": {"automated_events": wrong}}}, path)

    def test_non_target_cannot_be_finalized_as_measurable(self):
        for realization in ("other", "uncertain", "b"):
            with self.subTest(realization=realization), self.assertRaisesRegex(SystemExit, "target p/b"):
                self.finalize(annotation={**self.annotation, "acoustic_realization": realization})

    def test_unmeasurable_non_target_stays_in_ledger(self):
        result = self.finalize(annotation={**self.annotation, "status": "unmeasurable", "acoustic_realization": "other", "selected_candidate_rank": ""})
        self.assertIsNone(result["events"][0]["selected_release_time_s"])
        self.assertEqual(result["summary"]["status_counts"], {"unmeasurable": 1})

    def test_stale_key_and_duplicate_runner_ids_are_rejected(self):
        with self.assertRaisesRegex(SystemExit, "source_events_sha256"):
            self.finalize(source_hash="stale")
        with self.assertRaisesRegex(SystemExit, "duplicate runner_event_id"):
            self.finalize(duplicate=True)

    def test_key_anchor_and_candidate_integrity(self):
        variants = [
            ({**self.key_row, "anchor_s": 1.1}, "key anchor"),
            ({**self.key_row, "anchor_s": float("nan")}, "key anchor"),
            ({**self.key_row, "candidates": [{"time_s": 1.12}]}, "match source event"),
            ({**self.key_row, "candidates": [{"time_s": 1.02, "relative_ms": 120.0}]}, "relative time"),
            ({**self.key_row, "candidates": [{"time_s": float("nan")}]}, "outside review window"),
        ]
        for row, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(SystemExit, message):
                self.finalize(key_row=row)

    def test_manual_release_must_be_within_observed_clip(self):
        for relative in ("-2000", "350", "12345"):
            with self.subTest(relative=relative), self.assertRaisesRegex(SystemExit, "outside review window"):
                self.finalize(annotation={**self.annotation, "selected_candidate_rank": "", "selected_release_relative_ms": relative})
        for relative in ("nan", "inf"):
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "finite"):
                self.finalize(annotation={**self.annotation, "selected_candidate_rank": "", "selected_release_relative_ms": relative})
        result = self.finalize(annotation={**self.annotation, "selected_candidate_rank": "", "selected_release_relative_ms": "80"})
        self.assertAlmostEqual(result["events"][0]["selected_release_time_s"], 1.08)

    def test_unknown_legacy_review_bounds_require_regeneration(self):
        row = {key: value for key, value in self.key_row.items() if key != "review_window"}
        with self.assertRaisesRegex(SystemExit, "regenerate review materials"):
            self.finalize(key_row=row)

    def test_review_preparation_keeps_candidate_less_events(self):
        self.source["events"][0]["measurement"]["acoustic"]["candidates"] = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, events, output = root / "source.mp4", root / "events.json", root / "review"
            video.write_bytes(b"synthetic-test-placeholder")
            events.write_text(json.dumps(self.source))
            audio = SimpleNamespace(start_s=0.55, end_s=1.35, sample_rate=48000, waveform=np.zeros(38400), coverage_mask=np.ones(38400, dtype=bool))
            argv = ["prepare", "--video", str(video), "--events", str(events), "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(PREPARE, "decode_audio_window", return_value=audio), patch.object(PREPARE, "render_sheet", return_value=Image.new("RGB", (8, 8))), redirect_stdout(StringIO()):
                PREPARE.main()
            key = json.loads((output / "blind_key.json").read_text())
            self.assertEqual(key["selected_count"], 1)
            self.assertEqual(key["events"][0]["candidates"], [])
            self.assertEqual(key["events"][0]["review_window"], self.key_row["review_window"])

    def test_downstream_rejects_release_in_recorded_audio_hole(self):
        annotation = {"acoustic_realization": "p", "review_window": {
            "start_s": 0.0, "end_s": 0.7,
            "covered_intervals": [{"start_s": 0.0, "end_s": 0.2}, {"start_s": 0.3, "end_s": 0.5}],
        }}
        for time_s in (0.2, 0.25, 0.5, 0.65):
            with self.subTest(time_s=time_s):
                self.assertIn("acoustic_release_outside_decoded_coverage", acoustic_annotation_guard(annotation, "p", time_s)[1])
        self.assertEqual(acoustic_annotation_guard(annotation, "p", 0.3)[1], [])
        for invalid in (False, "0", 10**1000):
            bad = {"review_window": {"start_s": 0, "end_s": invalid}}
            with self.subTest(invalid_type=type(invalid).__name__):
                self.assertIn("invalid_acoustic_review_window", acoustic_annotation_guard(bad, "p", 0.1)[1])

    def test_legacy_omission_disclosed_but_explicit_nested_mismatch_rejected(self):
        attribution, reasons = acoustic_annotation_guard({}, "p", 1.0)
        self.assertEqual((attribution, reasons), ("legacy_unspecified", []))
        for annotation in ({"acoustic_realization": "other"}, {"annotation": {"acoustic_realization": "uncertain"}}, {"acoustic_realization": "p", "annotation": {"acoustic_realization": "b"}}):
            self.assertIn("acoustic_realization_not_target", acoustic_annotation_guard(annotation, "p", 1.0)[1])


if __name__ == "__main__":
    unittest.main()
