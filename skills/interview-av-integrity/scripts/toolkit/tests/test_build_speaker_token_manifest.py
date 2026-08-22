from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_speaker_token_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("build_speaker_token_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SpeakerTokenManifestConfigurationTests(unittest.TestCase):
    def _write_references(self, payload: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "references.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_references_are_loaded_from_arbitrary_speaker_ids(self) -> None:
        path = self._write_references(
            {
                "subject_alpha": {
                    "display_name": "Subject A",
                    "time_s": 12.5,
                    "group": "candidate",
                },
                "peer_omega": {
                    "display_name": "Peer O",
                    "time_s": 24.0,
                    "group": "control",
                },
            }
        )
        references = MODULE._load_references(path)
        self.assertEqual(references["subject_alpha"]["group"], "candidate")
        self.assertEqual(references["peer_omega"]["group"], "control")
        self.assertEqual(references["peer_omega"]["time_s"], 24.0)

    def test_reference_configuration_requires_both_analysis_groups(self) -> None:
        path = self._write_references(
            {
                "one": {
                    "display_name": "One",
                    "time_s": 1.0,
                    "group": "candidate",
                },
                "two": {
                    "display_name": "Two",
                    "time_s": 2.0,
                    "group": "exclude",
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "missing: control"):
            MODULE._load_references(path)

    def test_interval_groups_come_from_reference_configuration(self) -> None:
        samples = [
            {
                "time_s": 0.0,
                "speaker_id": "subject_alpha",
                "best_score": 0.9,
                "margin": 0.5,
            },
            {
                "time_s": 0.5,
                "speaker_id": "peer_omega",
                "best_score": 0.8,
                "margin": 0.4,
            },
            {
                "time_s": 1.0,
                "speaker_id": "unknown",
                "best_score": 0.1,
                "margin": 0.0,
            },
        ]
        intervals = MODULE._compress_intervals(
            samples,
            start_s=0.0,
            end_s=1.5,
            sample_hz=2.0,
            speaker_groups={
                "subject_alpha": "candidate",
                "peer_omega": "control",
            },
        )
        self.assertEqual(
            [interval["group"] for interval in intervals],
            ["candidate", "control", "exclude"],
        )

    def test_omitted_end_is_resolved_from_media_duration(self) -> None:
        self.assertEqual(
            MODULE._resolve_analysis_end(None, media_duration_s=87.25),
            87.25,
        )
        self.assertEqual(
            MODULE._resolve_analysis_end(50.0, media_duration_s=87.25),
            50.0,
        )
        with self.assertRaisesRegex(ValueError, "exceeds media duration"):
            MODULE._resolve_analysis_end(90.0, media_duration_s=87.25)

    def test_references_argument_is_required_and_end_defaults_to_none(self) -> None:
        references = self._write_references(
            {
                "subject_alpha": {
                    "display_name": "Subject A",
                    "time_s": 1.0,
                    "group": "candidate",
                },
                "peer_omega": {
                    "display_name": "Peer O",
                    "time_s": 2.0,
                    "group": "control",
                },
            }
        )
        args = MODULE.parse_args(
            [
                "video.mp4",
                "words.json",
                "output",
                "--references-json",
                str(references),
            ]
        )
        self.assertIsNone(args.end)
        self.assertEqual(args.references_json, references)

    def test_complete_reading_inventory_fails_closed_and_uses_readings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.json"
            missing.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "words": [
                                    {"word": "場所", "start": 1.0, "end": 1.4}
                                ]
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires a reading"):
                MODULE._load_tokens(
                    missing,
                    start_s=0.0,
                    end_s=2.0,
                    require_complete_readings=True,
                )

            complete = root / "complete.json"
            complete.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "words": [
                                    {
                                        "word": "場所",
                                        "reading": "ばしょ",
                                        "start": 1.0,
                                        "end": 1.4,
                                    }
                                ]
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tokens, audit = MODULE._load_tokens(
                complete,
                start_s=0.0,
                end_s=2.0,
                require_complete_readings=True,
            )
        self.assertEqual(tokens[0]["kana"], "ば")
        self.assertEqual(tokens[0]["word"], "場所")
        self.assertEqual(audit["token_inventory_scope"], "reading_complete")


if __name__ == "__main__":
    unittest.main()
