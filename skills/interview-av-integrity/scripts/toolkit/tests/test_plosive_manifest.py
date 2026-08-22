from __future__ import annotations

import unittest

from video_integrity_analyzer.plosive_manifest import (
    SpeakerInterval,
    extract_bilabial_tokens,
    load_speaker_intervals,
)


class PlosiveManifestTests(unittest.TestCase):
    def test_all_target_kana_are_returned_with_exclusions(self) -> None:
        whisper = {
            "segments": [
                {
                    "id": 7,
                    "no_speech_prob": 0.01,
                    "words": [
                        {"word": "ンプ", "start": 2.0, "end": 2.4, "probability": 0.99},
                        {"word": "ば", "start": 9.8, "end": 10.0, "probability": 0.40},
                    ],
                }
            ]
        }
        intervals = [SpeakerInterval("e1", "s1", "candidate", 0.0, 12.0)]
        events = extract_bilabial_tokens(
            whisper,
            intervals,
            interview_end_s=20.0,
            boundary_guard_s=0.5,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].phoneme_class, "p")
        self.assertAlmostEqual(events[0].anchor_s, 2.3)
        self.assertTrue(events[0].eligible)
        self.assertFalse(events[1].eligible)
        self.assertIn("low_asr_probability", str(events[1].exclusion_reason))

    def test_boundary_guard_prevents_switch_contamination(self) -> None:
        whisper = {
            "segments": [
                {
                    "words": [
                        {"word": "パ", "start": 5.1, "end": 5.2, "probability": 1.0}
                    ]
                }
            ]
        }
        intervals = [SpeakerInterval("e1", "s1", "control", 5.0, 10.0)]
        event = extract_bilabial_tokens(
            whisper,
            intervals,
            interview_end_s=20.0,
            boundary_guard_s=0.5,
        )[0]
        self.assertFalse(event.eligible)
        self.assertEqual(event.exclusion_reason, "near_active_speaker_transition")

    def test_word_reading_exposes_bilabial_hidden_by_kanji_surface(self) -> None:
        whisper = {
            "segments": [
                {
                    "words": [
                        {
                            "word": "場所",
                            "reading": "ばしょ",
                            "start": 2.0,
                            "end": 2.4,
                            "probability": 0.99,
                        }
                    ]
                }
            ]
        }
        intervals = [SpeakerInterval("e1", "s1", "candidate", 0.0, 12.0)]
        event = extract_bilabial_tokens(
            whisper,
            intervals,
            interview_end_s=20.0,
            boundary_guard_s=0.5,
        )[0]
        self.assertEqual(event.phoneme_class, "b")
        self.assertEqual(event.token_text, "場所")
        self.assertEqual(event.reading_text, "ばしょ")
        self.assertEqual(event.reading_source, "reading")

    def test_interval_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_speaker_intervals(
                [
                    {"epoch_id": "a", "speaker": "a", "group": "candidate", "start_s": 0, "end_s": 2},
                    {"epoch_id": "b", "speaker": "b", "group": "control", "start_s": 1, "end_s": 3},
                ]
            )


if __name__ == "__main__":
    unittest.main()
