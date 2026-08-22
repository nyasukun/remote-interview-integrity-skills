from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_closure_evidence_video.py"
SPEC = importlib.util.spec_from_file_location("verify_closure_evidence_video", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PREVIEW_SCRIPT = Path(__file__).resolve().parents[2] / "render_layout_preview.py"
PREVIEW_SPEC = importlib.util.spec_from_file_location(
    "av_integrity_render_layout_preview", PREVIEW_SCRIPT
)
assert PREVIEW_SPEC is not None and PREVIEW_SPEC.loader is not None
PREVIEW_MODULE = importlib.util.module_from_spec(PREVIEW_SPEC)
sys.modules[PREVIEW_SPEC.name] = PREVIEW_MODULE
PREVIEW_SPEC.loader.exec_module(PREVIEW_MODULE)

from video_integrity_analyzer.layout_reference import (  # noqa: E402
    REQUIRED_QUALITY_CHECKS,
    load_approved_layout_reference,
    sha256_file,
    verify_approved_preview,
)

FPS = 24
REPEAT = 4
NORMAL_FRAMES = 24
SLOW_FRAMES = NORMAL_FRAMES * REPEAT
INTRO_FRAMES = 12
GAP_FRAMES = 3
OUTRO_FRAMES = 8
EVENTS = (
    ("novel-alpha", 7.25, "closure_absent"),
    ("reference.omega", 42.875, "contact_reference"),
)
LIMITATION = (
    "本資料はA/V不整合の原因、発話者の本人性、国籍、所属、意図を"
    "判定するものではありません。"
)


def _row(output_index: int, **values: object) -> dict[str, str]:
    row = {
        "output_frame_index": str(output_index),
        "output_pts_s": f"{output_index / FPS:.9f}",
        "case_order": "",
        "event_id": "",
        "phase": "",
        "source_pts_s": "",
        "source_frame_index": "",
    }
    row.update({key: str(value) for key, value in values.items()})
    return row


def _synthetic_manifest_and_rows() -> tuple[dict, list[dict[str, str]]]:
    cases: list[dict] = []
    rows: list[dict[str, str]] = []
    cursor = 0
    for index in range(INTRO_FRAMES):
        rows.append(_row(index, phase="intro"))
    cursor += INTRO_FRAMES

    for order, (event_id, release, classification) in enumerate(EVENTS, start=1):
        source_start = release - 0.5
        source_end = release + 0.5
        normal_start = cursor
        slow_start = normal_start + NORMAL_FRAMES
        cases.append({
            "order": order,
            "event_id": event_id,
            "classification": classification,
            "presentation_label": classification,
            "label": event_id,
            "note": "manifest-driven synthetic case",
            "source_release_s": release,
            "mouth_crop_normalized_xyxy": [0.2, 0.4, 0.6, 0.7],
            "mouth_crop_xywh": [384, 432, 768, 324],
            "normal": {
                "source_start_s": source_start,
                "source_end_s": source_end,
                "output_start_s": normal_start / FPS,
                "output_end_s": (normal_start + NORMAL_FRAMES) / FPS,
                "speed": 1.0,
                "marker_output_s": normal_start / FPS + 0.5,
                "output_frame_count": NORMAL_FRAMES,
            },
            "slow": {
                "source_start_s": source_start,
                "source_end_s": source_end,
                "output_start_s": slow_start / FPS,
                "output_end_s": (slow_start + SLOW_FRAMES) / FPS,
                "speed": 1 / REPEAT,
                "marker_output_s": slow_start / FPS + 0.5 * REPEAT,
                "output_frame_count": SLOW_FRAMES,
            },
        })
        source_base = round(source_start * FPS)
        native = [
            (source_start + offset / FPS, source_base + offset)
            for offset in range(NORMAL_FRAMES)
        ]
        for offset, (source_pts, source_index) in enumerate(native):
            rows.append(_row(
                normal_start + offset,
                case_order=order,
                event_id=event_id,
                phase="normal",
                source_pts_s=f"{source_pts:.9f}",
                source_frame_index=source_index,
            ))
        for offset, (source_pts, source_index) in enumerate(native):
            for repeat_offset in range(REPEAT):
                rows.append(_row(
                    slow_start + offset * REPEAT + repeat_offset,
                    case_order=order,
                    event_id=event_id,
                    phase="slow",
                    source_pts_s=f"{source_pts:.9f}",
                    source_frame_index=source_index,
                ))
        gap_start = slow_start + SLOW_FRAMES
        for gap_offset in range(GAP_FRAMES):
            rows.append(_row(
                gap_start + gap_offset,
                case_order=order,
                event_id=event_id,
                phase="gap",
                source_pts_s=f"{native[-1][0]:.9f}",
                source_frame_index=native[-1][1],
            ))
        cursor = gap_start + GAP_FRAMES

    for offset in range(OUTRO_FRAMES):
        rows.append(_row(cursor + offset, phase="outro"))
    cursor += OUTRO_FRAMES
    manifest = {
        "schema_version": 1,
        "render": {
            "width": 1920,
            "height": 1080,
            "fps": FPS,
            "audio_sample_rate": 48_000,
            "normal_source_frames_per_output_frame": 1,
            "slow_source_frame_repeat": REPEAT,
            "video_interpolation": False,
            "source_frame_timing": {
                "is_cfr": True,
                "expected_fps": FPS,
                "decoded_frame_count": 100,
                "outlier_interval_count": 0,
            },
            "limitation": LIMITATION,
            "layout": {"mouth_inset_xyxy": [1396, 98, 1882, 414]},
        },
        "cards": {
            "intro_frames": INTRO_FRAMES,
            "intro_duration_s": INTRO_FRAMES / FPS,
            "event_gap_frames": GAP_FRAMES,
            "event_gap_duration_s": GAP_FRAMES / FPS,
            "outro_frames": OUTRO_FRAMES,
            "outro_duration_s": OUTRO_FRAMES / FPS,
        },
        "classification_counts": {
            "closure_absent": 1,
            "contact_reference": 1,
            "sync_reference": 0,
        },
        "cases": cases,
        "output": {
            "total_frame_count": cursor,
            "expected_duration_s": cursor / FPS,
        },
    }
    return manifest, rows


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_frame_map(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(MODULE.FRAME_MAP_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_mp4(path: Path, frame_count: int = 3) -> None:
    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=FPS)
    video.width = 1920
    video.height = 1080
    video.pix_fmt = "yuv420p"
    video.options = {"preset": "ultrafast", "crf": "35"}
    audio = container.add_stream("aac", rate=48_000)
    audio.layout = "stereo"
    audio_pts = 0
    for index in range(frame_count):
        frame = av.VideoFrame(1920, 1080, "yuv420p")
        for plane, value in zip(frame.planes, (16, 128, 128)):
            plane.update(bytes([value]) * plane.buffer_size)
        frame.pts = index
        frame.time_base = Fraction(1, FPS)
        for packet in video.encode(frame):
            container.mux(packet)
        sample_count = 2000
        t = (np.arange(sample_count) + audio_pts) / 48_000
        waveform = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(np.stack((waveform, waveform))),
            format="fltp",
            layout="stereo",
        )
        audio_frame.sample_rate = 48_000
        audio_frame.pts = audio_pts
        audio_frame.time_base = Fraction(1, 48_000)
        audio_pts += sample_count
        for packet in audio.encode(audio_frame):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()


class ClosureEvidenceVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._layout_temporary = tempfile.TemporaryDirectory()
        root = Path(cls._layout_temporary.name)
        cls.input_manifest = root / "input_event_manifest.json"
        cls.input_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "title": "Synthetic layout-gate fixture",
                    "events": [
                        {
                            "event_id": "fixture-layout-event",
                            "source_release_s": 2.0,
                            "classification": "closure_absent",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        preview = root / "preview.png"
        review_sheet = root / "review_sheet.png"
        provenance = root / "preview_provenance.json"
        PREVIEW_MODULE.render_layout_preview(
            cls.input_manifest,
            preview,
            review_sheet=review_sheet,
            report=provenance,
        )
        payload = json.loads(cls.input_manifest.read_text(encoding="utf-8"))
        payload["layout_review"] = {
            "status": "APPROVED",
            "approval_basis": "synthetic test fixture explicitly approved",
            "preview": {"path": str(preview), "sha256": sha256_file(preview)},
            "preview_provenance": {
                "path": str(provenance),
                "sha256": sha256_file(provenance),
            },
            "review_sheet": {
                "path": str(review_sheet),
                "sha256": sha256_file(review_sheet),
            },
            "quality_checks": {
                name: True for name in REQUIRED_QUALITY_CHECKS
            },
        }
        cls.input_manifest.write_text(json.dumps(payload), encoding="utf-8")
        cls.layout_reference = load_approved_layout_reference()
        cls.layout_review = verify_approved_preview(
            cls.input_manifest, cls.layout_reference
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._layout_temporary.cleanup()

    def _synthetic_manifest_and_rows(self) -> tuple[dict, list[dict[str, str]]]:
        manifest, rows = _synthetic_manifest_and_rows()
        manifest["input_event_manifest"] = str(self.input_manifest)
        manifest["render"]["layout_reference"] = copy.deepcopy(
            self.layout_reference
        )
        manifest["render"]["layout_review"] = copy.deepcopy(self.layout_review)
        return manifest, rows

    def _verify_manifest(self, manifest: dict):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            _write_manifest(path, manifest)
            return MODULE.verify_manifest(path)

    def test_novel_two_case_manifest_and_frame_map_pass(self) -> None:
        manifest, rows = self._synthetic_manifest_and_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            map_path = root / "map.csv"
            _write_manifest(manifest_path, manifest)
            _write_frame_map(map_path, rows)
            loaded, manifest_metrics = MODULE.verify_manifest(manifest_path)
            map_metrics = MODULE.verify_frame_map(map_path, loaded)
        self.assertEqual(manifest_metrics["case_count"], 2)
        self.assertEqual(manifest_metrics["event_order"], ["novel-alpha", "reference.omega"])
        self.assertEqual(map_metrics["row_count"], manifest["output"]["total_frame_count"])
        self.assertEqual(map_metrics["normal_frame_rows"], 2 * NORMAL_FRAMES)
        self.assertEqual(map_metrics["slow_frame_rows"], 2 * SLOW_FRAMES)

    def test_duplicate_event_id_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["cases"][1]["event_id"] = manifest["cases"][0]["event_id"]
        with self.assertRaisesRegex(MODULE.VerificationError, "not unique"):
            self._verify_manifest(manifest)

    def test_non_chronological_source_order_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["cases"][1]["source_release_s"] = 1.0
        with self.assertRaisesRegex(MODULE.VerificationError, "chronological"):
            self._verify_manifest(manifest)

    def test_non_sequential_order_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["cases"][1]["order"] = 7
        with self.assertRaisesRegex(MODULE.VerificationError, "order must"):
            self._verify_manifest(manifest)

    def test_out_of_bounds_mouth_roi_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["cases"][0]["mouth_crop_normalized_xyxy"] = [0.2, 0.4, 1.2, 0.7]
        with self.assertRaisesRegex(MODULE.VerificationError, "mouth ROI"):
            self._verify_manifest(manifest)

    def test_marker_formula_change_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["cases"][0]["slow"]["marker_output_s"] += 0.01
        with self.assertRaisesRegex(MODULE.VerificationError, "marker"):
            self._verify_manifest(manifest)

    def test_classification_count_mismatch_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["classification_counts"]["closure_absent"] = 2
        with self.assertRaisesRegex(MODULE.VerificationError, "actual"):
            self._verify_manifest(manifest)

    def test_incomplete_limitation_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["render"]["limitation"] = "原因だけは判定しません。"
        with self.assertRaisesRegex(MODULE.VerificationError, "identity"):
            self._verify_manifest(manifest)

    def test_unverified_source_cfr_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["render"]["source_frame_timing"]["is_cfr"] = False
        with self.assertRaisesRegex(MODULE.VerificationError, "strict CFR"):
            self._verify_manifest(manifest)

    def test_layout_reference_hash_tampering_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["render"]["layout_reference"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.VerificationError, "layout_reference"):
            self._verify_manifest(manifest)

    def test_missing_or_tampered_layout_review_is_rejected(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        del manifest["render"]["layout_review"]
        with self.assertRaisesRegex(MODULE.VerificationError, "layout_review"):
            self._verify_manifest(manifest)

        manifest, _ = self._synthetic_manifest_and_rows()
        manifest["render"]["layout_review"]["preview"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(MODULE.VerificationError, "layout_review"):
            self._verify_manifest(manifest)

    def test_non_declared_slow_hold_is_rejected(self) -> None:
        manifest, rows = self._synthetic_manifest_and_rows()
        slow_rows = [
            row for row in rows
            if row["event_id"] == "novel-alpha" and row["phase"] == "slow"
        ]
        slow_rows[3]["source_frame_index"] = slow_rows[4]["source_frame_index"]
        slow_rows[3]["source_pts_s"] = slow_rows[4]["source_pts_s"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.csv"
            _write_frame_map(path, rows)
            with self.assertRaisesRegex(MODULE.VerificationError, "exactly 4"):
                MODULE.verify_frame_map(path, manifest)

    def test_gap_hold_change_is_rejected(self) -> None:
        manifest, rows = self._synthetic_manifest_and_rows()
        gap = next(row for row in rows
                   if row["event_id"] == "novel-alpha" and row["phase"] == "gap")
        gap["source_frame_index"] = str(int(gap["source_frame_index"]) - 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.csv"
            _write_frame_map(path, rows)
            with self.assertRaisesRegex(MODULE.VerificationError, "gap"):
                MODULE.verify_frame_map(path, manifest)

    def test_spec_compliant_synthetic_mp4_fully_decodes(self) -> None:
        manifest, _ = self._synthetic_manifest_and_rows()
        manifest = copy.deepcopy(manifest)
        manifest["output"] = {"total_frame_count": 3, "expected_duration_s": 3 / FPS}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.mp4"
            _write_synthetic_mp4(path, frame_count=3)
            metrics = MODULE.verify_media(path, manifest)
        self.assertEqual(metrics["video_codec"], "h264")
        self.assertEqual(metrics["audio_codec"], "aac")
        self.assertEqual(metrics["decoded_video_frames"], 3)
        self.assertLess(metrics["av_end_difference_s"], 0.05)

    def test_reports_include_hashes_and_failure_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: dict[str, Path] = {}
            hashes: dict[str, str] = {}
            for name in ("manifest", "frame_map", "video"):
                path = root / name
                path.write_bytes(name.encode())
                paths[name] = path
                hashes[name] = MODULE.sha256_file(path)
            checks = [MODULE.CheckResult("synthetic", False, {}, "expected failure")]
            markdown, result_json = MODULE.write_reports(
                root, paths=paths, hashes=hashes, checks=checks,
            )
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            markdown_text = markdown.read_text(encoding="utf-8")
        self.assertEqual(payload["status"], "FAIL")
        self.assertIn(hashes["video"], markdown_text)


if __name__ == "__main__":
    unittest.main()
