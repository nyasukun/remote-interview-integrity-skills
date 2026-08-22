from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parents[1]))

from acoustics.extract_acoustic_features import extract  # noqa: E402
from common import ManifestError, load_manifest, sha256  # noqa: E402
from render import render  # noqa: E402
from verify import VerificationError, _contact_sheet_targets, verify  # noqa: E402
from render_layout_preview import render_preview  # noqa: E402


def write_audio(path: Path, seconds: float = 2.4, rate: int = 48_000) -> None:
    count = int(round(seconds * rate))
    time = np.arange(count) / rate
    steps = np.asarray([120.0, 150.0, 185.0, 220.0, 260.0])
    frequency = steps[(np.floor(time / 0.24).astype(int) % len(steps))]
    phase = 2 * np.pi * np.cumsum(frequency) / rate
    envelope = np.minimum(1.0, np.minimum(time / 0.01, (seconds - time) / 0.01))
    left = 0.24 * np.sin(phase) * envelope
    right = 0.20 * np.sin(phase + 0.02) * envelope
    pcm = np.clip(np.round(np.stack((left, right), axis=1) * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def clip_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_type": "non_biometric_descriptive_acoustic_features",
        "groups": [
            {"group_id": "focus_anchor", "role": "designated_point", "group_label": "Designated segment", "short_label": "Designated", "display_color_hex": "#FF9120"},
            {"group_id": "comparison_one", "role": "comparison_reference", "group_label": "Reference one", "short_label": "Ref 1", "display_color_hex": "#2EA0FF"},
            {"group_id": "comparison_two", "role": "comparison_reference", "group_label": "Reference two", "short_label": "Ref 2", "display_color_hex": "#46CDA6"},
        ],
        "designated_group_id": "focus_anchor",
        "comparison_group_ids": ["comparison_one", "comparison_two"],
        "clips": [
            {"clip_id": "segment-anchor", "group_id": "focus_anchor", "clip_label": "Anchor", "language_label": "caller supplied", "label_origin": "caller supplied; not audio inferred", "source": "fixture.wav", "start_s": 0.02, "end_s": 0.27},
            {"clip_id": "segment-r1a", "group_id": "comparison_one", "clip_label": "Reference 1A", "language_label": "caller supplied", "label_origin": "caller supplied; not audio inferred", "source": "fixture.wav", "start_s": 0.52, "end_s": 0.77},
            {"clip_id": "segment-r1b", "group_id": "comparison_one", "clip_label": "Reference 1B", "language_label": "caller supplied", "label_origin": "caller supplied; not audio inferred", "source": "fixture.wav", "start_s": 1.02, "end_s": 1.27},
            {"clip_id": "segment-r2a", "group_id": "comparison_two", "clip_label": "Reference 2A", "language_label": "caller supplied", "label_origin": "caller supplied; not audio inferred", "source": "fixture.wav", "start_s": 1.52, "end_s": 1.77},
        ],
    }


def render_manifest_payload(root: Path, clip_manifest: Path, provenance_hash: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "title": "Synthetic designated-anchor comparison",
        "limitation_text": "Descriptive signal differences only. No identity determination.",
        "clip_manifest": {"path": clip_manifest.name, "sha256": sha256(clip_manifest)},
        "feature_artifacts": {},
        "comparison": {
            "anchor_group": "focus_anchor",
            "display_anchor_group": "focus_anchor",
            "point_group": "focus_anchor",
            "reference_groups": ["comparison_one", "comparison_two"],
            "delta_definition": "group metric minus user-designated point metric",
        },
        "timeline": {"intro_seconds": 0.25, "gap_seconds": 0.05, "outro_seconds": 0.25, "edge_fade_ms": 5.0},
        "output": {"width": 1920, "height": 1080, "fps": 24, "audio_rate": 48000},
        "provenance_manifests": [{"role": "synthetic_selection", "path": "selection.json", "sha256": provenance_hash}],
    }


def attach_dummy_acoustic_artifacts(root: Path, payload: dict[str, object]) -> None:
    artifact_manifest = root / "dummy_artifact_manifest.json"
    acoustic_features = root / "dummy_acoustic_features.json"
    artifact_manifest.write_text("{}\n", encoding="utf-8")
    acoustic_features.write_text("{}\n", encoding="utf-8")
    payload["feature_artifacts"] = {
        "artifact_manifest": {"path": artifact_manifest.name, "sha256": sha256(artifact_manifest)},
        "acoustic_features": {"path": acoustic_features.name, "sha256": sha256(acoustic_features)},
    }


def write_render_manifest(root: Path, provenance: Path) -> Path:
    clip_payload = clip_manifest_payload()
    acoustic_input = root / "clip_manifest.json"
    acoustic_input.write_text(json.dumps(clip_payload, indent=2), encoding="utf-8")
    acoustic_dir = root / "authoritative_acoustics"
    extract(acoustic_input, acoustic_dir, sample_rate=48_000)
    artifact_manifest = acoustic_dir / "artifact_manifest.json"
    acoustic_features = acoustic_dir / "acoustic_features.json"
    payload = render_manifest_payload(root, acoustic_input, sha256(provenance))
    payload["feature_artifacts"] = {
        "artifact_manifest": {"path": str(artifact_manifest.relative_to(root)), "sha256": sha256(artifact_manifest)},
        "acoustic_features": {"path": str(acoustic_features.relative_to(root)), "sha256": sha256(acoustic_features)},
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


class RenderingSmokeTests(unittest.TestCase):
    def test_contact_sheet_targets_stay_inside_short_intro_and_outro(self) -> None:
        clips = [
            {
                "clip_id": "case-a",
                "output_start_sample": 9_600,
                "output_end_sample_exclusive": 33_600,
            }
        ]
        targets = dict(_contact_sheet_targets(clips, duration=0.9))
        self.assertGreater(targets["intro"], 0.0)
        self.assertLess(targets["intro"], 0.2)
        self.assertGreater(targets["outro"], 0.7)
        self.assertLess(targets["outro"], 0.9)

    def test_local_layout_preview_uses_generic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text('{"selection": "synthetic"}\n', encoding="utf-8")
            manifest = write_render_manifest(root, provenance)
            preview = render_preview(manifest, root / "preview.png")
            self.assertTrue(preview.is_file())
            from PIL import Image

            with Image.open(preview) as image:
                self.assertEqual(image.size, (1920, 1080))

    def test_loader_accepts_one_through_three_caller_labeled_comparison_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text("{}\n", encoding="utf-8")
            base = clip_manifest_payload()
            for count in (1, 2, 3):
                with self.subTest(comparison_group_count=count):
                    clip_payload = json.loads(json.dumps(base))
                    if count == 1:
                        clip_payload["groups"] = clip_payload["groups"][:2]
                        clip_payload["clips"] = [clip for clip in clip_payload["clips"] if clip["group_id"] != "comparison_two"]
                        clip_payload["comparison_group_ids"] = ["comparison_one"]
                    elif count == 3:
                        clip_payload["groups"].append({"group_id": "comparison_three", "role": "comparison_reference", "group_label": "Caller label three", "short_label": "Ref 3", "display_color_hex": "#B17EFF"})
                        clip_payload["clips"].append({"clip_id": "segment-r3a", "group_id": "comparison_three", "clip_label": "Reference 3A", "language_label": "caller supplied", "label_origin": "caller supplied; not audio inferred", "source": "fixture.wav", "start_s": 2.02, "end_s": 2.27})
                        clip_payload["comparison_group_ids"].append("comparison_three")
                    clip_manifest = root / f"clips_{count}.json"
                    clip_manifest.write_text(json.dumps(clip_payload), encoding="utf-8")
                    payload = render_manifest_payload(root, clip_manifest, sha256(provenance))
                    payload["comparison"]["reference_groups"] = list(clip_payload["comparison_group_ids"])
                    attach_dummy_acoustic_artifacts(root, payload)
                    manifest = root / f"manifest_{count}.json"
                    manifest.write_text(json.dumps(payload), encoding="utf-8")
                    spec = load_manifest(manifest)
                    self.assertEqual(len(spec.comparison_groups), count)

    def test_loader_rejects_anchor_that_does_not_resolve_to_designated_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text("{}\n", encoding="utf-8")
            clip_manifest = root / "clips.json"
            clip_manifest.write_text(json.dumps(clip_manifest_payload()), encoding="utf-8")
            payload = render_manifest_payload(root, clip_manifest, sha256(provenance))
            attach_dummy_acoustic_artifacts(root, payload)
            payload["comparison"]["anchor_group"] = "comparison_one"
            payload["comparison"]["display_anchor_group"] = "comparison_one"
            payload["comparison"]["point_group"] = "comparison_one"
            payload["comparison"]["reference_groups"] = ["focus_anchor", "comparison_two"]
            clip_payload = json.loads(clip_manifest.read_text(encoding="utf-8"))
            clip_payload["designated_group_id"] = "comparison_one"
            clip_payload["comparison_group_ids"] = ["focus_anchor", "comparison_two"]
            clip_manifest.write_text(json.dumps(clip_payload), encoding="utf-8")
            payload["clip_manifest"]["sha256"] = sha256(clip_manifest)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "group role does not match comparison configuration"):
                load_manifest(manifest)

    def test_generic_render_and_full_qa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text('{"selection": "synthetic"}\n', encoding="utf-8")
            manifest = write_render_manifest(root, provenance)
            video = root / "caller_chosen_video_name.mp4"
            effective = root / "caller_chosen_manifest_name.json"
            frames = root / "caller_chosen_frames_name.csv"
            audio = root / "caller_chosen_audio_name.csv"
            render(manifest, video, effective, frames, audio)
            result = verify(video, effective, frames, audio, root / "caller_chosen_qa_dir")
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(all(path.is_file() for path in (video, effective, frames, audio)))
            payload = json.loads(effective.read_text(encoding="utf-8"))
            self.assertEqual(payload["comparison"]["anchor_group"], "focus_anchor")
            self.assertEqual(payload["comparison"]["display_anchor_group"], "focus_anchor")
            self.assertEqual(payload["comparison"]["point_group"], "focus_anchor")
            self.assertFalse({"composite_score", "ranking", "winner", "identity_inference"} & set(payload["comparison"]))
            selection = next(item for item in payload["provenance_manifests"] if item["role"] == "synthetic_selection")
            self.assertEqual(selection["sha256"], sha256(provenance))
            self.assertEqual(payload["clip_manifest"]["sha256"], sha256(root / "clip_manifest.json"))
            self.assertEqual(payload["output"]["frame_map_sha256"], sha256(frames))
            self.assertEqual(payload["output"]["audio_map_sha256"], sha256(audio))
            self.assertLess(effective.stat().st_size, 262_144)
            for metric in payload["summary"]["metrics"].values():
                self.assertEqual(metric["signed_deltas_to_designated"]["focus_anchor"], {"min": 0.0, "max": 0.0, "median": 0.0})
            self.assertTrue(payload["authoritative_acoustic_match"]["all_summaries_exact_match"])
            for check in result["audio"]["clip_checks"]:
                self.assertGreaterEqual(check["normalized_correlation"], 0.97)

    def test_verifier_rejects_declared_display_anchor_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text("{}\n", encoding="utf-8")
            manifest = write_render_manifest(root, provenance)
            video, effective, frames, audio = (root / name for name in ("video.mp4", "effective.json", "frames.csv", "audio.csv"))
            render(manifest, video, effective, frames, audio)
            payload = json.loads(effective.read_text(encoding="utf-8"))
            payload["comparison"]["display_anchor_group"] = "comparison_one"
            tampered = root / "tampered.json"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "declared/display/point anchor mismatch"):
                verify(video, tampered, frames, audio, root / "tampered_qa")
            payload["comparison"]["display_anchor_group"] = payload["comparison"]["anchor_group"]
            payload["comparison"]["winner"] = False
            forbidden = root / "forbidden.json"
            forbidden.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "prohibited field exists"):
                verify(video, forbidden, frames, audio, root / "forbidden_qa")
            del payload["comparison"]["winner"]
            payload["clips"][0]["metrics"]["authoritative_summary"]["waveform"]["rms_dbfs"] += 1.0
            summary_tampered = root / "summary_tampered.json"
            summary_tampered.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "effective/authoritative summary mismatch"):
                verify(video, summary_tampered, frames, audio, root / "summary_tampered_qa")

    def test_verifier_rejects_recursive_identity_fields_and_unsafe_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text("{}\n", encoding="utf-8")
            manifest = write_render_manifest(root, provenance)
            video, effective, frames, audio = (
                root / name
                for name in ("video.mp4", "effective.json", "frames.csv", "audio.csv")
            )
            render(manifest, video, effective, frames, audio)
            original = json.loads(effective.read_text(encoding="utf-8"))
            mutations = (
                ("top", lambda value: value.__setitem__("same_speaker_probability", 0.9)),
                (
                    "comparison",
                    lambda value: value["comparison"].__setitem__(
                        "same_speaker_probability", 0.9
                    ),
                ),
                (
                    "summary",
                    lambda value: value["summary"].__setitem__(
                        "speaker_similarity", 0.9
                    ),
                ),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    payload = json.loads(json.dumps(original))
                    mutate(payload)
                    path = root / f"forbidden_{label}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(VerificationError, "prohibited field exists"):
                        verify(video, path, frames, audio, root / f"qa_{label}")

            unsafe = json.loads(json.dumps(original))
            unsafe["limitation"]["permanent_text"] = (
                "Descriptive signal differences only. No identity determination. "
                "The speaker is a confirmed match."
            )
            unsafe_path = root / "unsafe_limitation.json"
            unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "approved non-identity statement"):
                verify(video, unsafe_path, frames, audio, root / "qa_unsafe")

    def test_verifier_rejects_semantic_frame_and_audio_map_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "fixture.wav")
            provenance = root / "selection.json"
            provenance.write_text("{}\n", encoding="utf-8")
            manifest = write_render_manifest(root, provenance)
            video, effective, frames, audio = (
                root / name
                for name in ("video.mp4", "effective.json", "frames.csv", "audio.csv")
            )
            render(manifest, video, effective, frames, audio)
            original = json.loads(effective.read_text(encoding="utf-8"))

            with frames.open(encoding="utf-8", newline="") as handle:
                frame_rows = list(csv.DictReader(handle))
                frame_fields = list(frame_rows[0])
            clip_row = next(row for row in frame_rows if row["phase"] == "clip")
            clip_row["source_time_s"] = "999"
            changed_frames = root / "changed_frames.csv"
            with changed_frames.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=frame_fields)
                writer.writeheader()
                writer.writerows(frame_rows)
            payload = json.loads(json.dumps(original))
            payload["output"]["frame_map"] = str(changed_frames)
            payload["output"]["frame_map_sha256"] = sha256(changed_frames)
            changed_effective = root / "changed_frame_effective.json"
            changed_effective.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "source-time mismatch"):
                verify(video, changed_effective, changed_frames, audio, root / "qa_frame_time")

            with frames.open(encoding="utf-8", newline="") as handle:
                frame_rows = list(csv.DictReader(handle))
                frame_fields = list(frame_rows[0])
            swapped = next(row for row in frame_rows if row["clip_id"] == "segment-r1a")
            swapped["clip_id"] = "segment-r1b"
            swapped_frames = root / "swapped_frames.csv"
            with swapped_frames.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=frame_fields)
                writer.writeheader()
                writer.writerows(frame_rows)
            payload = json.loads(json.dumps(original))
            payload["output"]["frame_map"] = str(swapped_frames)
            payload["output"]["frame_map_sha256"] = sha256(swapped_frames)
            swapped_effective = root / "swapped_frame_effective.json"
            swapped_effective.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "frame map clip mismatch"):
                verify(video, swapped_effective, swapped_frames, audio, root / "qa_frame_swap")

            with audio.open(encoding="utf-8", newline="") as handle:
                audio_rows = list(csv.DictReader(handle))
                audio_fields = list(audio_rows[0])
            audio_clip = next(row for row in audio_rows if row["phase"] == "clip")
            audio_clip["source_start_s"] = str(float(audio_clip["source_start_s"]) + 1.0)
            changed_audio = root / "changed_audio.csv"
            with changed_audio.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=audio_fields)
                writer.writeheader()
                writer.writerows(audio_rows)
            payload = json.loads(json.dumps(original))
            payload["output"]["audio_map"] = str(changed_audio)
            payload["output"]["audio_map_sha256"] = sha256(changed_audio)
            changed_audio_effective = root / "changed_audio_effective.json"
            changed_audio_effective.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "source interval differs"):
                verify(video, changed_audio_effective, frames, changed_audio, root / "qa_audio")


if __name__ == "__main__":
    unittest.main()
