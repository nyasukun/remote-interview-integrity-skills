from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
from PIL import Image

from video_integrity_analyzer.plosive_sync import decode_audio_window


HELPERS = Path(__file__).resolve().parents[3] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HELPERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_blinded_acoustic_review")
FINALIZE = load("finalize_blinded_acoustic_review")


def write_synthetic_gapped_audio(path):
    """Two real PCM packets cover [0,.2) and [.3,.5), leaving a PTS hole."""
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("pcm_s16le", rate=48000)
        stream.layout = "mono"
        values = np.full((1, 9600), 1000, dtype=np.int16)
        for pts in (0, 14400):
            frame = av.AudioFrame.from_ndarray(values, format="s16", layout="mono")
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, 48000)
            frame.pts = pts
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


class AudioReviewCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audio-review-coverage-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "synthetic-gap.mkv"
        write_synthetic_gapped_audio(self.video)
        self.events = self.root / "events.json"

    def prepare(self, anchor=.3, before=.3, after=.4):
        self.events.write_text(json.dumps({"events": [{
            "selection": {"event_id": "synthetic-1", "anchor_s": anchor, "phoneme_class": "p", "eligible": True},
            "measurement": {"acoustic": {"candidates": [
                {"time_s": time_s} for time_s in (.1, .25, .35, .6)
            ]}},
        }]}))
        self.output = self.root / "review"
        argv = ["prepare", "--video", str(self.video), "--events", str(self.events),
                "--output-dir", str(self.output), "--window-before", str(before), "--window-after", str(after)]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()), patch.object(
            PREPARE, "render_sheet", return_value=Image.new("RGB", (8, 8))
        ) as render:
            PREPARE.main()
        self.render_call = render.call_args
        self.key_path = self.output / "blind_key.json"
        self.key = json.loads(self.key_path.read_text())
        return self.key["events"][0]

    def finalize(self, relative_ms="0", *, status="measurable", rank="", key=None):
        if key is not None:
            self.key_path.write_text(json.dumps(key))
        row = {
            "blind_id": "AR-0001", "status": status, "acoustic_realization": "p",
            "confidence": "high", "reason": "synthetic validation fixture",
            "selected_candidate_rank": rank, "selected_release_relative_ms": relative_ms,
        }
        annotations = self.output / "acoustic_annotations.csv"
        with annotations.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        result = self.root / "finalized.json"
        argv = ["finalize", "--events", str(self.events), "--blind-key", str(self.key_path),
                "--annotations", str(annotations), "--output", str(result)]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            FINALIZE.main()
        return json.loads(result.read_text())

    def test_real_decoder_preserves_requested_timeline_with_pts_hole_and_eof_padding(self):
        audio = decode_audio_window(self.video, start_s=0, end_s=.7, sample_rate=48000)
        self.assertEqual(len(audio.waveform), 33600)
        self.assertEqual(audio.end_s, .7)
        self.assertTrue(np.all(np.isnan(audio.waveform[9600:14400])))
        self.assertFalse(np.any(audio.coverage_mask[24000:]))
        self.assertEqual(PREPARE.covered_audio_intervals(
            audio.waveform, audio.coverage_mask, audio.start_s, audio.sample_rate
        ), [{"start_s": 0.0, "end_s": .2}, {"start_s": .3, "end_s": .5}])

    def test_preparation_keeps_actual_coverage_and_filters_hole_candidates(self):
        row = self.prepare()
        self.assertEqual(row["review_window"], {"start_s": 0.0, "end_s": .7, "covered_intervals": [
            {"start_s": 0.0, "end_s": .2}, {"start_s": .3, "end_s": .5},
        ]})
        self.assertEqual([item["time_s"] for item in row["candidates"]], [.1, .35])
        self.assertEqual(self.render_call.args[3:5], (0.0, .7))
        self.assertEqual(self.render_call.kwargs["covered_intervals"], row["review_window"]["covered_intervals"])

    def test_manual_times_in_hole_and_eof_padding_are_rejected(self):
        self.prepare()
        for relative in ("-100", "-50", "200", "250"):
            with self.subTest(relative=relative), self.assertRaisesRegex(SystemExit, "outside decoded audio coverage"):
                self.finalize(relative)
        result = self.finalize("50")
        self.assertEqual(result["events"][0]["selected_release_time_s"], .35)
        self.assertEqual(result["events"][0]["review_window"], self.key["events"][0]["review_window"])

    def test_rank_cannot_reintroduce_a_source_candidate_in_a_coverage_hole(self):
        self.prepare()
        key = copy.deepcopy(self.key)
        key["events"][0]["candidates"] = [{"rank": 1, "time_s": .25, "relative_ms": -50.0}]
        with self.assertRaisesRegex(SystemExit, "candidate outside decoded audio coverage"):
            self.finalize("", rank="1", key=key)

    def test_candidate_rank_metadata_must_match_list_position(self):
        self.prepare()
        variants = [[2, 1], [1, 1], [True, 2], [1.0, 2], ["1", 2], [None, 2]]
        for ranks in variants:
            key = copy.deepcopy(self.key)
            for candidate, rank in zip(key["events"][0]["candidates"], ranks):
                candidate["rank"] = rank
            with self.subTest(ranks=ranks), self.assertRaisesRegex(SystemExit, "candidate rank.*list position"):
                self.finalize("", rank="1", key=key)
        key = copy.deepcopy(self.key)
        for candidate in key["events"][0]["candidates"]:
            del candidate["rank"]
        result = self.finalize("", rank="2", key=key)
        self.assertEqual(result["events"][0]["selected_release_time_s"], .35)

    def test_all_uncovered_window_retained_only_as_unmeasurable(self):
        row = self.prepare(anchor=1.0, before=.1, after=.1)
        self.assertEqual(row["review_window"]["covered_intervals"], [])
        self.assertEqual(row["candidates"], [])
        with self.assertRaisesRegex(SystemExit, "outside decoded audio coverage"):
            self.finalize("0")
        result = self.finalize("", status="unmeasurable")
        self.assertIsNone(result["events"][0]["selected_release_time_s"])
        self.assertEqual(result["summary"]["status_counts"], {"unmeasurable": 1})

    def test_missing_or_invalid_coverage_requires_regeneration(self):
        self.prepare()
        variants = [None, [{"start_s": 0, "end_s": float("nan")}],
                    [{"start_s": False, "end_s": .2}],
                    [{"start_s": "0", "end_s": .2}],
                    [{"start_s": 0, "end_s": 10 ** 1000}],
                    [None],
                    [{"start_s": 0, "end_s": .4}, {"start_s": .3, "end_s": .5}],
                    [{"start_s": 0, "end_s": .8}]]
        for intervals in variants:
            key = copy.deepcopy(self.key)
            if intervals is None:
                key["schema_version"] = 2
                del key["events"][0]["review_window"]["covered_intervals"]
            else:
                key["events"][0]["review_window"]["covered_intervals"] = intervals
            with self.subTest(intervals=intervals), self.assertRaisesRegex(SystemExit, "covered_intervals; regenerate"):
                self.finalize("50", key=key)

    def test_window_bounds_reject_boolean_or_string_provenance(self):
        self.prepare()
        for bound in (False, "0", None):
            key = copy.deepcopy(self.key)
            key["events"][0]["review_window"]["start_s"] = bound
            with self.subTest(bound=bound), self.assertRaisesRegex(SystemExit, "invalid review_window"):
                self.finalize("50", key=key)

    def test_nonfinite_values_never_count_as_covered_even_if_mask_is_true(self):
        self.assertEqual(PREPARE.covered_audio_intervals(
            np.array([1., np.nan, np.inf, 2.]), np.ones(4, dtype=bool), 0.0, 10
        ), [{"start_s": 0.0, "end_s": .1}, {"start_s": .3, "end_s": .4}])


if __name__ == "__main__":
    unittest.main()
