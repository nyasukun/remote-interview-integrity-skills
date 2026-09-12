"""Adversarial waveform tests for uncertainty-preserving target attribution.

These are deterministic signal controls, not recordings of Japanese phonemes.
They test whether ambiguity is retained, not phonetic classification accuracy.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from video_integrity_analyzer.plosive_manifest import (
    SpeakerInterval,
    extract_bilabial_tokens,
)
from video_integrity_analyzer.plosive_sync import (
    AcousticReleaseConfig,
    AudioWindow,
    VisualReleaseConfig,
    estimate_acoustic_release,
    _continuous_release,
)


SAMPLE_RATE = 48_000
RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_plosive_sync.py"
SPEC = importlib.util.spec_from_file_location("acoustic_attribution_safety_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def burst_and_tones(onsets: tuple[float, ...], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times = np.arange(int(SAMPLE_RATE * 0.8), dtype=np.float64) / SAMPLE_RATE
    waveform = rng.normal(0.0, 2e-5, size=len(times))
    for index, onset in enumerate(onsets):
        burst = (times >= onset) & (times < onset + 0.003)
        waveform[burst] += rng.normal(0.0, 0.20, size=np.count_nonzero(burst))
        tone = (times >= onset + 0.006) & (
            (times < onset + 0.045) if len(onsets) > 1 else True
        )
        waveform[tone] += 0.10 * np.sin(2 * np.pi * (220 + 40 * index) * times[tone])
        waveform[tone] += 0.025 * np.sin(2 * np.pi * 2_500 * times[tone])
    return waveform


def weak_burst_with_continuous_tail() -> np.ndarray:
    rng = np.random.default_rng(11)
    times = np.arange(int(SAMPLE_RATE * 0.9), dtype=np.float64) / SAMPLE_RATE
    waveform = rng.normal(0.0, 2e-5, size=len(times))
    burst = (times >= 0.350) & (times < 0.353)
    waveform[burst] += rng.normal(0.0, 0.05, size=np.count_nonzero(burst))
    noise_tail = (times >= 0.353) & (times < 0.368)
    waveform[noise_tail] += rng.normal(0.0, 0.012, size=np.count_nonzero(noise_tail))
    voiced = times >= 0.368
    ramp = np.clip((times[voiced] - 0.368) / 0.010, 0.0, 1.0)
    waveform[voiced] += ramp * (
        0.30 * np.sin(2.0 * np.pi * 220.0 * times[voiced])
        + 0.06 * np.sin(2.0 * np.pi * 2_500.0 * times[voiced])
    )
    return waveform


class AcousticAttributionSafetyTests(unittest.TestCase):
    def estimate(self, waveform: np.ndarray, *, config=None, anchor=0.34, **windows):
        return estimate_acoustic_release(
            waveform, SAMPLE_RATE, window_start_s=0.0, anchor_time_s=anchor,
            config=config or AcousticReleaseConfig.conservative(), **windows,
        )

    def test_display_candidate_limit_cannot_turn_ambiguity_into_acceptance(self):
        waveform = burst_and_tones((0.300, 0.420), seed=271828)
        results = [
            self.estimate(
                waveform, anchor=0.310,
                config=AcousticReleaseConfig.conservative(maximum_candidates=limit),
                target_window_s=(0.250, 0.385), next_window_s=(0.385, 0.500),
            )
            for limit in (1, 8)
        ]
        for result in results:
            self.assertFalse(result.measurable)
            self.assertIsNone(result.release_time_s)

    def test_two_short_pulses_separated_by_quiet_remain_ambiguous_by_default(self):
        times = np.arange(int(SAMPLE_RATE * 0.8)) / SAMPLE_RATE
        for seed in (906, 907, 908):
            rng = np.random.default_rng(seed)
            waveform = rng.normal(0.0, 2e-5, len(times))
            for onset in (0.300, 0.335):
                pulse = (times >= onset) & (times < onset + 0.009)
                waveform[pulse] += rng.normal(0.0, 0.20, np.count_nonzero(pulse))
            # Each pulse is 9 ms long, with 26 ms of quiet between them.
            with self.subTest(seed=seed):
                result = self.estimate(waveform, anchor=0.330, target_window_s=(0.250, 0.400))
                self.assertFalse(result.measurable)
                self.assertIsNone(result.release_time_s)

    def test_onset_shared_by_overlapping_words_is_not_automatically_owned(self):
        waveform = burst_and_tones((0.350,), seed=314159)
        result = self.estimate(
            waveform, target_window_s=(0.300, 0.450), next_window_s=(0.310, 0.500)
        )
        self.assertFalse(result.measurable)
        self.assertIsNone(result.release_time_s)

    def test_later_shared_or_previous_word_peak_cannot_be_absorbed_as_continuation(self):
        waveform = weak_burst_with_continuous_tail()
        for neighbor in ("previous_window_s", "next_window_s"):
            with self.subTest(neighbor=neighbor):
                result = self.estimate(
                    waveform, target_window_s=(0.30, 0.60),
                    **{neighbor: (0.36, 0.60)},
                )
                self.assertFalse(result.measurable)
                self.assertIsNone(result.release_time_s)
                shared = [candidate for candidate in result.candidates if candidate.attribution == "shared_word_span"]
                self.assertTrue(shared)
                self.assertTrue(all(not candidate.same_event_as_selected for candidate in shared))

    def test_selected_earlier_onset_remains_available_with_one_review_candidate(self):
        waveform = weak_burst_with_continuous_tail()
        full = self.estimate(waveform, target_window_s=(0.30, 0.60))
        limited = self.estimate(
            waveform, target_window_s=(0.30, 0.60),
            config=AcousticReleaseConfig.conservative(maximum_candidates=1),
        )
        self.assertTrue(limited.measurable)
        self.assertEqual(limited.release_time_s, full.release_time_s)
        self.assertLess(abs(limited.release_time_s - 0.350), 0.006)
        self.assertEqual([candidate.time_s for candidate in limited.candidates], [limited.release_time_s])

    def test_low_frequency_precursor_does_not_replace_known_broadband_burst(self):
        times = np.arange(int(SAMPLE_RATE * 0.8)) / SAMPLE_RATE
        for precursor_s in (0.320, 0.328):
            rng = np.random.default_rng(513)
            waveform = rng.normal(0.0, 2e-5, len(times))
            low_envelope = np.clip((times - precursor_s) / 0.001, 0.0, 1.0)
            low_envelope *= np.clip((0.360 - times) / 0.010, 0.0, 1.0)
            waveform += 0.03 * low_envelope * np.sin(2 * np.pi * 220 * times)
            burst = (times >= 0.350) & (times < 0.353)
            waveform[burst] += rng.normal(0.0, 0.20, np.count_nonzero(burst))
            tone = times >= 0.356
            waveform[tone] += 0.10 * np.sin(2 * np.pi * 220 * times[tone])
            waveform[tone] += 0.025 * np.sin(2 * np.pi * 2_500 * times[tone])
            with self.subTest(precursor_s=precursor_s):
                result = self.estimate(waveform, target_window_s=(0.28, 0.50))
                self.assertLess(abs(result.candidate_time_s - 0.350), 0.006)
                # The burst may still fail the conservative onset gates; in
                # that case abstain rather than substitute the earlier LF rise.
                self.assertTrue(
                    result.release_time_s is None
                    or abs(result.release_time_s - 0.350) < 0.006
                )

    def test_absent_between_peak_samples_cannot_establish_continuity(self):
        result = self.estimate(burst_and_tones((0.350,), seed=314159))
        earlier = result.selected_candidate
        later = replace(earlier, time_s=earlier.time_s + 0.020)
        self.assertFalse(_continuous_release(
            np.empty(0), np.empty(0), earlier, later,
            continuity_db=10.0, skip_s=0.008,
        ))

    def test_one_burst_cannot_auto_measure_two_occurrences_in_the_same_word(self):
        transcript = {"segments": [{"words": [{
            "word": "パパ", "reading": "パパ", "start": 0.28, "end": 0.40,
            "probability": 0.99,
        }]}]}
        tokens = extract_bilabial_tokens(
            transcript, [SpeakerInterval("e", "s", "candidate", 0.0, 2.0)],
            interview_end_s=1.5, boundary_guard_s=0,
        )
        self.assertEqual(len(tokens), 2)  # Preserve both occurrences in the ledger.
        waveform = burst_and_tones((0.350,), seed=314159)
        audio = AudioWindow(
            0.0, SAMPLE_RATE, waveform, np.ones(len(waveform), dtype=bool), 1.0
        )
        with patch.object(RUNNER, "decode_audio_window", return_value=audio), patch.object(
            RUNNER, "extract_native_lip_samples", return_value=[]
        ):
            records = [
                RUNNER._measure_token(
                    token, video=Path("synthetic.mp4"), face_model=Path("unused.task"),
                    roi=None, interview_end_s=1.5,
                    acoustic_config=AcousticReleaseConfig.conservative(),
                    visual_config=VisualReleaseConfig(),
                )
                for token in tokens if token.eligible
            ]
        self.assertLessEqual(sum(record["acoustic_auto_accepted"] for record in records), 1)

    def test_invalid_target_and_neighbor_spans_cannot_confer_attribution(self):
        waveform = burst_and_tones((0.350,), seed=314159)
        invalid = ((-0.1, 0.5), (0.35, 0.35), (0.4, 0.3), (float("nan"), 0.5), (0.0, float("inf")))
        for name in ("target_window_s", "previous_window_s", "next_window_s"):
            for window in invalid:
                windows = {"target_window_s": (0.3, 0.5), name: window}
                with self.subTest(name=name, window=window), self.assertRaises(ValueError):
                    self.estimate(waveform, **windows)

    def test_invalid_attribution_controls_fail_instead_of_changing_acceptance(self):
        waveform = burst_and_tones((0.350,), seed=314159)
        overrides = [
            {name: value}
            for name in ("target_boundary_guard_ms", "same_event_window_ms")
            for value in (-1.0, float("nan"), float("inf"))
        ] + [{"maximum_candidates": value} for value in (0, -1)] + [{"same_event_continuity_db": -1.0}]
        for override in overrides:
            with self.subTest(override=override), self.assertRaises((ValueError, TypeError)):
                self.estimate(
                    waveform, config=AcousticReleaseConfig.conservative(**override),
                    target_window_s=(0.3, 0.5),
                )


if __name__ == "__main__":
    unittest.main()
