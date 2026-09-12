from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acoustics.extract_acoustic_features import load_manifest
from acoustics.signal_features import sha256_file


class InputManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "shared.wav").touch()
        (self.root / "override.wav").touch()
        self.manifest = self.root / "input.json"

    def load(self, payload: object):
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        return load_manifest(self.manifest)

    def payload(self) -> dict:
        return {
            "source": "shared.wav",
            "clips": [
                {"id": "clip-a", "group": "focus", "start_s": 0.25, "end_s": 0.75},
                {"id": "clip-b", "group": "reference", "start_s": 1.0},
            ],
        }

    def test_manifest_aliases_resolve_to_the_same_input_spec(self) -> None:
        compact = self.payload()
        compact["groups"] = {
            "focus": {"label": "Focus", "color": "#aabbcc"},
            "reference": {"label": "Reference", "color": "#123abc"},
        }
        legacy = self.load(compact)
        canonical = self.load({
            "source_video": {"path": "shared.wav"},
            "groups": [
                {"group_id": "focus", "group_label": "Focus", "display_color_hex": "#AABBCC"},
                {"group_id": "reference", "group_label": "Reference", "display_color_hex": "#123ABC"},
            ],
            "clips": [
                {"clip_id": "clip-a", "group_id": "focus", "start_s": 0.25, "end_s": 0.75},
                {"clip_id": "clip-b", "group_id": "reference", "start_s": 1.0},
            ],
        })
        self.assertEqual(legacy.groups, canonical.groups)
        self.assertEqual(legacy.clips, canonical.clips)
        self.assertEqual(canonical.manifest_path, self.manifest.resolve())
        self.assertEqual(canonical.manifest_sha256, sha256_file(self.manifest))

    def test_group_order_and_default_colors_follow_clip_discovery(self) -> None:
        payload = self.payload()
        payload["groups"] = [
            {"id": "unused"}, {"id": "reference"}, {"id": "focus"},
        ]
        result = self.load(payload)
        self.assertEqual(
            [(group.group_id, group.display_color_hex) for group in result.groups],
            [("focus", "#0072B2"), ("reference", "#D55E00")],
        )

    def test_clip_source_overrides_shared_source_and_retains_caller_metadata(self) -> None:
        payload = self.payload()
        payload["clips"][0].update({
            "source": {"path": "override.wav"},
            "label": "Chosen clip",
            "label_origin": "caller supplied",
            "language_label": "caller language",
            "selection_notes": {"overlap": False},
        })
        result = self.load(payload)
        first, second = result.clips
        self.assertEqual(first.path, (self.root / "override.wav").resolve())
        self.assertEqual(second.path, (self.root / "shared.wav").resolve())
        self.assertEqual(first.clip_label, "Chosen clip")
        self.assertEqual(first.label_origin, "caller supplied")
        self.assertEqual(first.metadata, {
            "language_label": "caller language", "selection_notes": {"overlap": False},
        })
        self.assertEqual((first.start_s, first.end_s), (0.25, 0.75))
        self.assertIsNone(second.end_s)

    def test_conflicting_group_metadata_is_rejected(self) -> None:
        for field, first_value, second_value, error in (
            ("group_label", "First label", "Second label", "inconsistent labels"),
            ("group_color_hex", "#123456", "#654321", "inconsistent colors"),
        ):
            with self.subTest(field=field):
                payload = self.payload()
                first, second = payload["clips"]
                second["group"] = first["group"]
                first[field], second[field] = first_value, second_value
                with self.assertRaisesRegex(ValueError, error):
                    self.load(payload)

    def test_invalid_clip_ids_and_intervals_are_rejected(self) -> None:
        for updates, error in (
            ({"id": "../clip"}, "invalid or duplicate clip id"),
            ({"id": "clip-b"}, "invalid or duplicate clip id"),
            ({"group": "bad/group"}, "group_id is invalid"),
            ({"start_s": -0.1}, "invalid interval"),
            ({"end_s": 0.25}, "invalid interval"),
            ({"start_s": "nan"}, "must be finite"),
            ({"end_s": "inf"}, "must be finite"),
            ({"start_s": "unavailable"}, "must be numeric"),
        ):
            with self.subTest(updates=updates):
                payload = self.payload()
                payload["clips"][0].update(updates)
                with self.assertRaisesRegex(ValueError, error):
                    self.load(payload)

    def test_missing_or_nonexistent_source_is_rejected(self) -> None:
        payload = self.payload()
        del payload["source"]
        with self.assertRaisesRegex(ValueError, "has no source path"):
            self.load(payload)
        payload["source"] = "missing.wav"
        with self.assertRaises(FileNotFoundError):
            self.load(payload)


if __name__ == "__main__":
    unittest.main()
