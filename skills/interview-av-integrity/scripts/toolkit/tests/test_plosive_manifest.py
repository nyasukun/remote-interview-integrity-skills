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

    def test_neighbouring_word_spans_follow_transcript_order_across_segments(self) -> None:
        whisper = {
            "segments": [
                {
                    "words": [
                        {"word": "パ", "start": 1.0, "end": 1.2, "probability": 0.99},
                        {"word": "と", "start": 1.2, "end": 1.4, "probability": 0.99},
                    ]
                },
                {
                    "words": [
                        {"word": "ブ", "start": 2.0, "end": 2.3, "probability": 0.99},
                        {"word": "リ", "start": "bad", "end": 2.5, "probability": 0.99},
                        {"word": "ポ", "start": 2.5, "end": 2.7, "probability": 0.99},
                    ]
                },
            ]
        }
        intervals = [SpeakerInterval("e1", "s1", "candidate", 0.0, 12.0)]
        events = extract_bilabial_tokens(
            whisper, intervals, interview_end_s=20.0, boundary_guard_s=0.0
        )
        self.assertEqual([event.kana for event in events], ["パ", "ブ", "ポ"])
        first, second, third = events
        self.assertIsNone(first.previous_word_window_s)
        self.assertEqual(first.next_word_window_s, (1.2, 1.4))
        # Neighbour spans cross the segment boundary and the pause.
        self.assertEqual(second.previous_word_window_s, (1.2, 1.4))
        # A neighbour with unusable timestamps is disclosed as None.
        self.assertIsNone(second.next_word_window_s)
        self.assertIsNone(third.previous_word_window_s)
        self.assertIsNone(third.next_word_window_s)
        self.assertEqual(first.as_dict()["next_word_window_s"], (1.2, 1.4))
        for event in events:
            self.assertEqual(event.word_occurrence_index, 0)
            self.assertEqual(event.word_occurrence_anchors_s, (event.anchor_s,))

    def test_multiple_bilabial_kana_in_one_word_share_occurrence_anchors(self) -> None:
        whisper = {
            "segments": [
                {
                    "words": [
                        {"word": "パン", "start": 0.0, "end": 0.2, "probability": 0.99},
                        {"word": "バックアップ", "start": 1.0, "end": 1.6, "probability": 0.99},
                        {"word": "ク", "start": 1.6, "end": 1.6, "probability": 0.99},
                    ]
                }
            ]
        }
        intervals = [SpeakerInterval("e1", "s1", "candidate", 0.0, 12.0)]
        events = extract_bilabial_tokens(
            whisper, intervals, interview_end_s=20.0, boundary_guard_s=0.0
        )
        self.assertEqual([event.kana for event in events], ["パ", "バ", "プ"])
        pa, ba, pu = events
        self.assertEqual((pa.word_occurrence_index, pa.word_occurrence_anchors_s), (0, (pa.anchor_s,)))
        self.assertEqual(ba.word_occurrence_index, 0)
        self.assertEqual(pu.word_occurrence_index, 1)
        self.assertEqual(ba.word_occurrence_anchors_s, (ba.anchor_s, pu.anchor_s))
        self.assertEqual(pu.word_occurrence_anchors_s, ba.word_occurrence_anchors_s)
        self.assertLess(ba.anchor_s, pu.anchor_s)
        # A zero-length neighbouring word cannot bound attribution.
        self.assertIsNone(pu.next_word_window_s)
        self.assertEqual(ba.previous_word_window_s, (0.0, 0.2))

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
