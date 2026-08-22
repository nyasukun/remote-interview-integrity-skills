from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_closure_evidence_video.py"
SPEC = importlib.util.spec_from_file_location("render_closure_evidence_video", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ClosureEvidenceRendererTests(unittest.TestCase):
    def test_manifest_aliases_and_pixel_roi_are_accepted(self) -> None:
        payload = {
            "name": "test",
            "defaults": {"before_s": 0.7, "after_s": 0.9},
            "cases": [
                {
                    "runner_event_id": "pb-1",
                    "audio_release_time_s": 12.5,
                    "classification": "no_visible_contact",
                    "audio_context": "コンポーネント",
                    "mouth_bbox": [614.4, 388.8, 1305.6, 982.8],
                },
                {
                    "id": "pb-2",
                    "burst_time_s": 20.0,
                    "kind": "sync",
                    "label": "プロダクト",
                    "window": {"start": 19.2, "end": 20.8},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            manifest = MODULE.load_manifest(path)
        self.assertEqual(manifest.title, "test")
        self.assertEqual([event.category for event in manifest.events], ["missing", "reference"])
        self.assertAlmostEqual(manifest.events[0].pre_s, 0.7)
        self.assertAlmostEqual(manifest.events[0].mouth_roi[0], 0.32)
        self.assertAlmostEqual(manifest.events[1].post_s, 0.8)

    def test_manifest_is_presented_in_source_chronology_not_class_blocks(self) -> None:
        payload = {
            "events": [
                {"id": "late-red", "release_s": 30.0, "category": "missing", "order": 1},
                {"id": "early-green", "release_s": 10.0, "category": "reference", "order": 3},
                {"id": "middle-red", "release_s": 20.0, "category": "missing", "order": 2},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = MODULE.load_manifest(path)
        self.assertEqual(
            [event.event_id for event in manifest.events],
            ["early-green", "middle-red", "late-red"],
        )

    def test_duplicate_manifest_event_ids_are_rejected(self) -> None:
        payload = {
            "events": [
                {"id": "arbitrary-id", "release_s": 1.0, "category": "missing"},
                {"id": "arbitrary-id", "release_s": 2.0, "category": "reference"},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                MODULE.load_manifest(path)

    def test_late_contact_note_discloses_possible_following_phoneme(self) -> None:
        event = MODULE.parse_event(
            {
                "id": "pb-late",
                "release_s": 10.0,
                "category": "missing",
                "visual_evidence": {
                    "first_clear_contact_offset_ms_through_plus500": 197.7,
                    "summary": "initial summary",
                },
            },
            1,
            {"pre_s": 0.8, "post_s": 0.8, "mouth_roi": None},
        )
        self.assertIn("+197.7 ms", event.note)
        self.assertIn("後続音素の口形である可能性", event.note)

    def test_optional_warning_is_data_driven_not_event_id_driven(self) -> None:
        event = MODULE.parse_event(
            {
                "id": "novel-warning-id",
                "release_s": 3.0,
                "category": "missing",
                "presentation_warning": "語彙意図と表層音が一致しない可能性",
            },
            1,
            {"pre_s": 0.8, "post_s": 0.8, "mouth_roi": None},
        )
        self.assertEqual(event.warning, "語彙意図と表層音が一致しない可能性")

    def test_source_release_schema_is_accepted(self) -> None:
        event = MODULE.parse_event(
            {
                "event_id": "novel-source-release",
                "source_release_s": 12.345,
                "classification": "closure_absent",
                "label": "target",
                "note": "no visible contact",
            },
            1,
            {"pre_s": 0.8, "post_s": 0.8, "mouth_roi": None},
        )
        self.assertEqual(event.event_id, "novel-source-release")
        self.assertAlmostEqual(event.release_s, 12.345)
        self.assertEqual(event.category, "missing")

    def test_manifest_phone_label_is_configurable(self) -> None:
        payload = {
            "phone_label": "/b/",
            "events": [
                {
                    "event_id": "novel-phone-label",
                    "source_release_s": 12.345,
                    "classification": "sync_reference",
                    "label": "target",
                    "note": "visible contact",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = MODULE.load_manifest(path)
        self.assertEqual(manifest.phone_label, "/b/")
        self.assertEqual(manifest.events[0].reference_kind, "sync")
        self.assertEqual(manifest.events[0].category_ja, "同期参照（閉鎖あり）")

    def test_contact_reference_is_not_presented_as_synchronized(self) -> None:
        event = MODULE.parse_event(
            {
                "event_id": "contact-only",
                "source_release_s": 12.345,
                "classification": "contact_reference",
            },
            1,
            {"pre_s": 0.8, "post_s": 0.8, "mouth_roi": None},
        )
        self.assertEqual(event.category_ja, "閉鎖あり参照")
        self.assertEqual(event.classification, "contact_reference")

    def test_card_defaults_are_five_seconds(self) -> None:
        args = MODULE.build_parser().parse_args(
            ["--manifest", "m.json", "--output", "out.mp4"]
        )
        self.assertEqual(args.intro_seconds, 5.0)
        self.assertEqual(args.outro_seconds, 5.0)

    def test_native_frame_selection_is_consecutive_and_unique(self) -> None:
        image = Image.new("RGB", (2, 2))
        event = MODULE.EvidenceEvent(
            event_id="pb",
            release_s=10.5,
            category="missing",
            label="test",
            note="test",
            pre_s=0.5,
            post_s=0.5,
            mouth_roi=MODULE.DEFAULT_MOUTH_ROI,
            order=1,
        )
        frames = [
            MODULE.SourceFrame(9.958333333 + index / 24.0, 239 + index, image)
            for index in range(28)
        ]
        selected = MODULE.select_native_frames(frames, event)
        indices = [frame.source_frame_index for frame in selected]
        self.assertEqual(len(indices), 24)
        self.assertEqual(indices, list(range(indices[0], indices[0] + 24)))
        slow = [frame.source_frame_index for frame in selected for _ in range(4)]
        self.assertTrue(all(slow[index:index + 4] == [value] * 4 for index, value in zip(range(0, len(slow), 4), indices)))

    def test_audio_stretch_has_exact_target_and_fades_edges(self) -> None:
        audio = np.ones((2, 48_000), dtype=np.float32)
        stretched = MODULE.stretch_audio(audio, 192_000)
        self.assertEqual(stretched.shape, (2, 192_000))
        self.assertAlmostEqual(float(stretched[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(stretched[0, -1]), 0.0, places=6)
        self.assertGreater(float(stretched[0, 10_000]), 0.99)

    def test_point_eight_window_is_quantized_without_audio_speed_change(self) -> None:
        event = MODULE.EvidenceEvent(
            event_id="pb",
            release_s=10.0,
            category="missing",
            label="test",
            note="test",
            pre_s=0.8,
            post_s=0.8,
            mouth_roi=MODULE.DEFAULT_MOUTH_ROI,
            order=1,
        )
        quantized = MODULE.quantize_event_window(event)
        self.assertAlmostEqual(quantized.pre_s, 19 / 24)
        self.assertAlmostEqual(quantized.post_s, 19 / 24)
        self.assertEqual(round((quantized.pre_s + quantized.post_s) * 24), 38)

    def test_effective_manifest_records_actual_windows(self) -> None:
        event = MODULE.EvidenceEvent(
            event_id="pb",
            release_s=10.0,
            category="reference",
            label="test",
            note="test",
            pre_s=0.8,
            post_s=0.8,
            mouth_roi=MODULE.DEFAULT_MOUTH_ROI,
            order=1,
        )
        manifest = MODULE.Manifest("title", "subtitle", None, (event,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "effective.json"
            MODULE.write_effective_manifest(
                output,
                input_manifest=root / "input.json",
                source_video=root / "source.mp4",
                output_video=root / "output.mp4",
                frame_map=root / "map.csv",
                manifest=manifest,
                requested_pre_s=0.8,
                requested_post_s=0.8,
                intro_seconds=2.0,
                outro_seconds=2.0,
                event_gap_seconds=0.25,
                source_frame_timing={"is_cfr": True, "expected_fps": 24.0},
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["render"]["slow_source_frame_repeat"], 4)
        self.assertFalse(payload["render"]["video_interpolation"])
        self.assertTrue(payload["render"]["source_frame_timing"]["is_cfr"])
        self.assertEqual(payload["phone_label"], "/p/")
        self.assertEqual(
            payload["render"]["layout"]["mouth_inset_xyxy"],
            list(MODULE.MOUTH_INSET_XYXY),
        )
        self.assertEqual(payload["cases"][0]["normal"]["source_start_s"], 9.2)
        self.assertEqual(payload["cases"][0]["classification"], "contact_reference")
        self.assertGreater(payload["output"]["expected_duration_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
