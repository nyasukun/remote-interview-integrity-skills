from __future__ import annotations

import math
import unittest

import numpy as np

from video_integrity_analyzer.plosive_sync import (
    AcousticReleaseCandidate,
    AcousticReleaseConfig,
    LipApertureSample,
    attribute_release_candidates,
    combine_release_estimates,
    estimate_acoustic_release,
    estimate_visual_release,
    identify_mouth_shape_at_acoustic_release,
    normalized_lip_aperture,
)


def synthetic_plosive(
    *, sample_rate: int = 48_000, release_s: float = 0.350
) -> np.ndarray:
    rng = np.random.default_rng(314159)
    times = np.arange(int(sample_rate * 0.8), dtype=np.float64) / sample_rate
    waveform = rng.normal(0.0, 2e-5, size=len(times))
    burst = (times >= release_s) & (times < release_s + 0.003)
    waveform[burst] += rng.normal(0.0, 0.20, size=np.count_nonzero(burst))
    vowel = times >= release_s + 0.006
    waveform[vowel] += 0.10 * np.sin(2.0 * np.pi * 220.0 * times[vowel])
    waveform[vowel] += 0.025 * np.sin(2.0 * np.pi * 2_500.0 * times[vowel])
    return waveform


def synthetic_double_plosive(*, sample_rate: int = 48_000) -> np.ndarray:
    rng = np.random.default_rng(271828)
    times = np.arange(int(sample_rate * 0.8), dtype=np.float64) / sample_rate
    waveform = rng.normal(0.0, 2e-5, size=len(times))
    for release_s, fundamental_hz in ((0.300, 220.0), (0.420, 260.0)):
        burst = (times >= release_s) & (times < release_s + 0.003)
        waveform[burst] += rng.normal(0.0, 0.20, size=np.count_nonzero(burst))
        vowel = (times >= release_s + 0.006) & (times < release_s + 0.045)
        waveform[vowel] += 0.10 * np.sin(
            2.0 * np.pi * fundamental_hz * times[vowel]
        )
        waveform[vowel] += 0.025 * np.sin(
            2.0 * np.pi * 2_500.0 * times[vowel]
        )
    return waveform


def synthetic_release_sequence(
    releases: list[tuple[float, float, float, float | None, float]],
    *,
    sample_rate: int = 48_000,
    duration_s: float = 0.9,
    seed: int = 7,
) -> np.ndarray:
    """Known-onset bursts, each followed 6 ms later by a vowel.

    ``releases`` holds ``(release_s, burst, vowel, vowel_end_s, f0_hz)``.
    """

    rng = np.random.default_rng(seed)
    times = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    waveform = rng.normal(0.0, 2e-5, size=len(times))
    for release_s, burst, vowel, vowel_end_s, f0_hz in releases:
        mask = (times >= release_s) & (times < release_s + 0.003)
        waveform[mask] += rng.normal(0.0, burst, size=np.count_nonzero(mask))
        voiced = times >= release_s + 0.006
        if vowel_end_s is not None:
            voiced &= times < vowel_end_s
        waveform[voiced] += vowel * np.sin(2.0 * np.pi * f0_hz * times[voiced])
        waveform[voiced] += 0.25 * vowel * np.sin(2.0 * np.pi * 2_500.0 * times[voiced])
    return waveform


def synthetic_weak_burst_then_loud_vowel(
    *, sample_rate: int = 48_000, release_s: float = 0.350
) -> np.ndarray:
    """A quiet /p/ burst, 15 ms of aspiration, then a much louder vowel."""

    rng = np.random.default_rng(11)
    times = np.arange(int(sample_rate * 0.9), dtype=np.float64) / sample_rate
    waveform = rng.normal(0.0, 2e-5, size=len(times))
    burst = (times >= release_s) & (times < release_s + 0.003)
    waveform[burst] += rng.normal(0.0, 0.05, size=np.count_nonzero(burst))
    aspiration = (times >= release_s + 0.003) & (times < release_s + 0.018)
    waveform[aspiration] += rng.normal(0.0, 0.012, size=np.count_nonzero(aspiration))
    voiced = times >= release_s + 0.018
    ramp = np.clip((times[voiced] - (release_s + 0.018)) / 0.010, 0.0, 1.0)
    waveform[voiced] += ramp * (
        0.30 * np.sin(2.0 * np.pi * 220.0 * times[voiced])
        + 0.06 * np.sin(2.0 * np.pi * 2_500.0 * times[voiced])
    )
    return waveform


def aperture_sequence(
    *,
    fps: float = 24.0,
    closed_indices: range = range(7, 10),
    repeated_indices: set[int] | None = None,
) -> list[LipApertureSample]:
    repeated_indices = repeated_indices or set()
    output: list[LipApertureSample] = []
    for index in range(20):
        aperture = 0.010 if index in closed_indices else 0.050
        output.append(
            LipApertureSample(
                time_s=index / fps,
                median_aperture=aperture,
                maximum_aperture=aperture + 0.002,
                aperture_spread=0.002,
                mouth_width_px=110.0,
                face_quality=0.90,
                pair_apertures=(aperture - 0.001, aperture, aperture + 0.002),
                repeated_frame=index in repeated_indices,
                repeat_distance=0.0 if index in repeated_indices else 0.02,
            )
        )
    return output


class LipGeometryTests(unittest.TestCase):
    def test_aperture_is_invariant_to_roll_scale_and_translation(self) -> None:
        points = np.zeros((478, 2), dtype=np.float64)
        points[61] = (-1.0, 0.0)
        points[291] = (1.0, 0.0)
        for (upper, lower), normalized_distance in zip(
            ((13, 14), (82, 87), (312, 317)), (0.010, 0.012, 0.014)
        ):
            pixel_distance = normalized_distance * 2.0
            points[upper] = (0.0, -pixel_distance / 2.0)
            points[lower] = (0.0, pixel_distance / 2.0)

        angle = math.radians(37.0)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        )
        transformed = points @ rotation.T * 3.7 + np.asarray((41.0, -8.0))
        geometry = normalized_lip_aperture(transformed)

        np.testing.assert_allclose(geometry.pair_apertures, (0.010, 0.012, 0.014))
        self.assertAlmostEqual(geometry.median_aperture, 0.012)
        self.assertAlmostEqual(geometry.aperture_spread, 0.004)


class AcousticReleaseTests(unittest.TestCase):
    def test_refines_asr_anchor_to_broadband_release(self) -> None:
        sample_rate = 48_000
        estimate = estimate_acoustic_release(
            synthetic_plosive(sample_rate=sample_rate),
            sample_rate,
            window_start_s=0.0,
            anchor_time_s=0.340,
            phone_class="p",
        )
        self.assertTrue(estimate.measurable)
        self.assertEqual(estimate.confidence, "high")
        self.assertIsNotNone(estimate.release_time_s)
        self.assertLess(abs(float(estimate.release_time_s) - 0.350), 0.004)
        self.assertGreater(estimate.selected_candidate.spectral_flux_z, 4.0)  # type: ignore[union-attr]

    def test_conservative_mode_reports_features_and_margin(self) -> None:
        estimate = estimate_acoustic_release(
            synthetic_plosive(),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.340,
            config=AcousticReleaseConfig.conservative(),
        )
        self.assertTrue(estimate.measurable)
        self.assertEqual(estimate.acceptance_mode, "conservative")
        self.assertTrue(
            estimate.runner_up_margin is None or estimate.runner_up_margin >= 1.5
        )
        candidate = estimate.selected_candidate
        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate.energy_rise_db, 10.0)  # type: ignore[union-attr]
        self.assertGreaterEqual(candidate.energy_slope_12ms_db, 10.0)  # type: ignore[union-attr]
        self.assertGreaterEqual(candidate.high_frequency_rise_db, 8.0)  # type: ignore[union-attr]
        self.assertGreaterEqual(candidate.score, 5.5)  # type: ignore[union-attr]

    def test_conservative_mode_rejects_ambiguous_top_candidate(self) -> None:
        estimate = estimate_acoustic_release(
            synthetic_double_plosive(),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.310,
            config=AcousticReleaseConfig.conservative(),
        )
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.release_time_s)
        self.assertIn("ambiguous_acoustic_top_candidate", estimate.exclusion_reasons)
        self.assertGreater(len(estimate.candidates), 1)
        self.assertLess(float(estimate.runner_up_margin), 1.5)

    def test_wide_search_retains_but_gate_rejects_far_candidate(self) -> None:
        estimate = estimate_acoustic_release(
            synthetic_plosive(release_s=0.300),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.100,
            config=AcousticReleaseConfig.conservative(),
        )
        self.assertIsNotNone(estimate.candidate_time_s)
        self.assertGreater(
            estimate.selected_candidate.distance_from_anchor_ms, 160.0  # type: ignore[union-attr]
        )
        self.assertFalse(estimate.measurable)
        self.assertIn(
            "acoustic_candidate_outside_conservative_anchor_gate",
            estimate.exclusion_reasons,
        )

    def test_permissive_mode_remains_available_for_diagnostics(self) -> None:
        estimate = estimate_acoustic_release(
            synthetic_plosive(),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.340,
            config=AcousticReleaseConfig(
                conservative_minimum_score=1_000.0,
                conservative_minimum_runner_up_margin=1_000.0,
            ),
        )
        self.assertTrue(estimate.measurable)
        self.assertEqual(estimate.acceptance_mode, "permissive")

    def test_flat_audio_is_not_reported_as_a_release(self) -> None:
        estimate = estimate_acoustic_release(
            np.zeros(24_000, dtype=np.float64),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.250,
            phone_class="b",
        )
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.release_time_s)
        self.assertTrue(estimate.exclusion_reasons)

    def test_missing_audio_coverage_is_disclosed(self) -> None:
        waveform = synthetic_plosive()
        coverage = np.ones(len(waveform), dtype=bool)
        coverage[int(0.33 * 48_000) : int(0.37 * 48_000)] = False
        estimate = estimate_acoustic_release(
            waveform,
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.340,
            coverage_mask=coverage,
        )
        self.assertFalse(estimate.measurable)
        self.assertIn("incomplete_audio_coverage", estimate.exclusion_reasons)


class AcousticBurstEvidenceTests(unittest.TestCase):
    """Constructed signals distinguish transient evidence from a tonal rise.

    These controls test candidate selection, not Japanese phoneme recognition.
    """

    SAMPLE_RATE = 48_000

    def test_harmonic_only_ramps_cannot_supply_a_release(self) -> None:
        times = np.arange(int(self.SAMPLE_RATE * 0.9)) / self.SAMPLE_RATE
        for mode in ("permissive", "conservative"):
            for ramp_ms in (1, 5, 10, 20, 30):
                rng = np.random.default_rng(20260913)
                waveform = rng.normal(0.0, 2e-5, len(times))
                ramp = np.clip((times - 0.350) / (ramp_ms / 1000.0), 0.0, 1.0)
                waveform += ramp * (
                    0.30 * np.sin(2 * np.pi * 220 * times)
                    + 0.06 * np.sin(2 * np.pi * 2_500 * times)
                )
                with self.subTest(mode=mode, ramp_ms=ramp_ms):
                    estimate = estimate_acoustic_release(
                        waveform,
                        self.SAMPLE_RATE,
                        window_start_s=0.0,
                        anchor_time_s=0.360,
                        phone_class="p",
                        config=AcousticReleaseConfig(acceptance_mode=mode),
                        target_window_s=(0.10, 0.80),
                    )
                    self.assertFalse(estimate.measurable)
                    self.assertIsNone(estimate.release_time_s)
                    self.assertIsNone(estimate.candidate_time_s)
                    self.assertIsNone(estimate.selected_candidate)
                    self.assertTrue(estimate.candidates)
                    self.assertIn(
                        "no_broadband_release_evidence", estimate.exclusion_reasons
                    )

    def test_louder_earlier_harmonic_rise_does_not_replace_later_burst(self) -> None:
        release_s = 0.450
        waveform = synthetic_release_sequence(
            [(release_s, 0.08, 0.09, None, 220.0)], seed=20260913
        )
        times = np.arange(len(waveform)) / self.SAMPLE_RATE
        earlier_envelope = np.clip((times - 0.300) / 0.010, 0.0, 1.0)
        earlier_envelope *= np.clip((0.385 - times) / 0.020, 0.0, 1.0)
        waveform += earlier_envelope * (
            0.30 * np.sin(2 * np.pi * 220 * times)
            + 0.06 * np.sin(2 * np.pi * 2_500 * times)
        )
        for mode in ("permissive", "conservative"):
            with self.subTest(mode=mode):
                estimate = estimate_acoustic_release(
                    waveform,
                    self.SAMPLE_RATE,
                    window_start_s=0.0,
                    anchor_time_s=0.440,
                    config=AcousticReleaseConfig(acceptance_mode=mode),
                    target_window_s=(0.20, 0.65),
                )
                self.assertTrue(estimate.measurable, estimate.exclusion_reasons)
                self.assertLess(abs(float(estimate.release_time_s) - release_s), 0.006)
                self.assertTrue(
                    any(abs(candidate.time_s - 0.300) < 0.010 for candidate in estimate.candidates)
                )

    def test_prevoicing_does_not_require_silence_before_a_burst(self) -> None:
        release_s = 0.350
        waveform = synthetic_plosive(release_s=release_s)
        times = np.arange(len(waveform)) / self.SAMPLE_RATE
        prevoicing = np.clip((times - 0.200) / 0.010, 0.0, 1.0)
        waveform += 0.005 * prevoicing * np.sin(2 * np.pi * 110 * times)
        for mode in ("permissive", "conservative"):
            with self.subTest(mode=mode):
                estimate = estimate_acoustic_release(
                    waveform,
                    self.SAMPLE_RATE,
                    window_start_s=0.0,
                    anchor_time_s=0.340,
                    phone_class="b",
                    config=AcousticReleaseConfig(acceptance_mode=mode),
                    target_window_s=(0.28, 0.50),
                )
                self.assertTrue(estimate.measurable, estimate.exclusion_reasons)
                self.assertLess(abs(float(estimate.release_time_s) - release_s), 0.006)

    def test_unobserved_burst_cannot_be_replaced_by_post_gap_tonal_onset(self) -> None:
        # The only injected noise transient is unobserved. Most of the review
        # window is covered, so aggregate coverage alone cannot protect its
        # onset boundary from being replaced by the following tonal rise.
        for replacement in ("masked", "nan", "inf"):
            waveform = synthetic_plosive()
            coverage = np.ones(len(waveform), dtype=bool)
            missing = slice(round(0.350 * self.SAMPLE_RATE), round(0.353 * self.SAMPLE_RATE))
            if replacement == "masked":
                coverage[missing] = False
            else:
                waveform[missing] = float(replacement)
            for mode in ("permissive", "conservative"):
                with self.subTest(replacement=replacement, mode=mode):
                    estimate = estimate_acoustic_release(
                        waveform,
                        self.SAMPLE_RATE,
                        window_start_s=0.0,
                        anchor_time_s=0.340,
                        coverage_mask=coverage,
                        config=AcousticReleaseConfig(acceptance_mode=mode),
                        target_window_s=(0.28, 0.50),
                    )
                    self.assertFalse(estimate.measurable)
                    self.assertIsNone(estimate.release_time_s)
                    self.assertIn("incomplete_audio_coverage", estimate.exclusion_reasons)

    def test_invalid_burst_controls_cannot_disable_evidence_screening(self) -> None:
        invalid_overrides = (
            {"burst_frame_ms": -4.0},
            {"burst_search_radius_ms": math.inf},
            {"burst_minimum_flatness": 0.0},
            {"burst_minimum_high_frequency_fraction": math.nan},
            {"burst_minimum_above_pre_db": True},
            {"burst_maximum_below_peak_db": "18"},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises((ValueError, TypeError)):
                estimate_acoustic_release(
                    synthetic_plosive(),
                    self.SAMPLE_RATE,
                    window_start_s=0.0,
                    anchor_time_s=0.340,
                    config=AcousticReleaseConfig(**override),
                )


class AcousticTargetAttributionTests(unittest.TestCase):
    """Constructed known-onset cases; they do not claim phoneme identity."""

    SAMPLE_RATE = 48_000
    TARGET = (0.30, 0.50)
    PREVIOUS = (0.10, 0.30)
    NEXT = (0.50, 0.60)

    def estimate(self, waveform: np.ndarray, anchor_s: float, **kwargs):
        return estimate_acoustic_release(
            waveform,
            self.SAMPLE_RATE,
            window_start_s=0.0,
            anchor_time_s=anchor_s,
            config=AcousticReleaseConfig.conservative(),
            **kwargs,
        )

    def test_weak_burst_followed_by_louder_vowel_is_one_event(self) -> None:
        # Regression: the burst and the voicing onset 18 ms later used to be
        # ranked as rival hypotheses and the event was rejected as ambiguous.
        estimate = self.estimate(synthetic_weak_burst_then_loud_vowel(), 0.340)
        self.assertTrue(estimate.measurable, estimate.exclusion_reasons)
        self.assertNotIn("ambiguous_acoustic_top_candidate", estimate.exclusion_reasons)
        self.assertLess(abs(float(estimate.release_time_s) - 0.350), 0.004)
        self.assertGreater(float(estimate.top_candidate_time_s), float(estimate.release_time_s))
        same_event = [c for c in estimate.candidates if c.same_event_as_selected]
        self.assertGreaterEqual(len(same_event), 2)
        self.assertTrue(all(c.attribution == "unbounded" for c in estimate.candidates))

    def test_stronger_onset_in_previous_word_does_not_displace_target(self) -> None:
        # Regression: an utterance onset 250 ms earlier scored within the
        # runner-up margin, so the correct in-word release was rejected.
        waveform = synthetic_release_sequence(
            [(0.150, 0.30, 0.20, 0.32, 220.0), (0.400, 0.15, 0.10, None, 220.0)]
        )
        estimate = self.estimate(
            waveform,
            0.380,
            target_window_s=self.TARGET,
            previous_window_s=self.PREVIOUS,
            next_window_s=self.NEXT,
        )
        self.assertTrue(estimate.measurable, estimate.exclusion_reasons)
        self.assertLess(abs(float(estimate.release_time_s) - 0.400), 0.004)
        self.assertEqual(estimate.target_window_s, self.TARGET)
        by_attribution = {c.attribution: c for c in estimate.candidates}
        self.assertIn("previous_word", by_attribution)
        self.assertLess(abs(by_attribution["previous_word"].time_s - 0.150), 0.004)
        self.assertEqual(estimate.candidates[0].attribution, "target")

    def test_boundary_hugging_neighbour_onset_is_rejected_not_reported(self) -> None:
        waveform = synthetic_release_sequence(
            [(0.275, 0.30, 0.20, 0.38, 220.0), (0.420, 0.10, 0.08, None, 220.0)]
        )
        estimate = self.estimate(
            waveform,
            0.400,
            target_window_s=self.TARGET,
            previous_window_s=self.PREVIOUS,
            next_window_s=self.NEXT,
        )
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.release_time_s)
        self.assertIn("ambiguous_target_attribution", estimate.exclusion_reasons)
        # The diagnostic candidate is the in-word one, never the neighbour.
        self.assertLess(abs(float(estimate.candidate_time_s) - 0.420), 0.004)
        contested = [c for c in estimate.candidates if c.attribution == "previous_word"]
        self.assertTrue(contested)
        self.assertLessEqual(float(contested[0].distance_from_target_window_ms), 40.0)

    def test_no_candidate_inside_target_word_reports_nothing(self) -> None:
        waveform = synthetic_release_sequence([(0.150, 0.30, 0.20, 0.32, 220.0)])
        estimate = self.estimate(
            waveform,
            0.380,
            target_window_s=self.TARGET,
            previous_window_s=self.PREVIOUS,
            next_window_s=self.NEXT,
        )
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.candidate_time_s)
        self.assertIsNone(estimate.selected_candidate)
        self.assertEqual(
            estimate.exclusion_reasons, ("no_acoustic_candidate_within_target_word",)
        )
        self.assertTrue(estimate.candidates)
        self.assertTrue(all(c.attribution == "previous_word" for c in estimate.candidates))

    def test_adjacent_tokens_are_not_assigned_the_same_release(self) -> None:
        waveform = synthetic_release_sequence(
            [(0.300, 0.20, 0.10, 0.345, 220.0), (0.420, 0.20, 0.10, 0.465, 260.0)]
        )
        first = self.estimate(
            waveform,
            0.305,
            target_window_s=(0.25, 0.36),
            previous_window_s=(0.10, 0.25),
            next_window_s=(0.36, 0.48),
        )
        second = self.estimate(
            waveform,
            0.420,
            phone_class="b",
            target_window_s=(0.36, 0.48),
            previous_window_s=(0.25, 0.36),
            next_window_s=(0.48, 0.60),
        )
        self.assertTrue(first.measurable, first.exclusion_reasons)
        self.assertTrue(second.measurable, second.exclusion_reasons)
        # 8 ms analysis frames localise a burst to within half a frame.
        self.assertLess(abs(float(first.release_time_s) - 0.300), 0.006)
        self.assertLess(abs(float(second.release_time_s) - 0.420), 0.006)

    def test_rough_anchor_shift_inside_word_span_still_resolves(self) -> None:
        waveform = synthetic_release_sequence([(0.400, 0.20, 0.10, None, 220.0)])
        estimate = self.estimate(
            waveform,
            0.280,
            target_window_s=self.TARGET,
            previous_window_s=self.PREVIOUS,
            next_window_s=self.NEXT,
        )
        self.assertTrue(estimate.measurable, estimate.exclusion_reasons)
        self.assertLess(abs(float(estimate.release_time_s) - 0.400), 0.006)

    def test_voicing_onset_inside_next_word_span_is_same_event(self) -> None:
        # Burst at the end of the target span; its voicing onset lands in the
        # next word's rough span and must not be treated as a rival.
        estimate = self.estimate(
            synthetic_weak_burst_then_loud_vowel(),
            0.340,
            target_window_s=(0.25, 0.36),
            previous_window_s=(0.10, 0.25),
            next_window_s=(0.36, 0.50),
        )
        self.assertTrue(estimate.measurable, estimate.exclusion_reasons)
        self.assertLess(abs(float(estimate.release_time_s) - 0.350), 0.004)
        continuation = [
            c for c in estimate.candidates if c.attribution == "next_word"
        ]
        self.assertTrue(continuation)
        self.assertTrue(all(c.same_event_as_selected for c in continuation))

    def test_attribution_labels_and_target_first_ordering(self) -> None:
        def candidate(time_s: float, score: float) -> AcousticReleaseCandidate:
            return AcousticReleaseCandidate(
                time_s=time_s,
                score=score,
                spectral_flux_z=0.0,
                high_frequency_rise_db=0.0,
                energy_rise_db=0.0,
                energy_slope_12ms_db=0.0,
                peak_rms_dbfs=-20.0,
                pre_rms_dbfs=-40.0,
                post_rms_dbfs=-20.0,
                local_coverage=1.0,
                distance_from_anchor_ms=0.0,
            )

        labelled = attribute_release_candidates(
            [
                candidate(0.29, 9.0),  # inside previous word
                candidate(0.35, 5.0),  # inside target
                candidate(0.62, 8.0),  # 20 ms past a gap after the target: nearest is target
                candidate(0.70, 7.0),  # 100 ms past the target, no next word: outside
                candidate(0.40, 6.0),  # inside target
            ],
            target_window_s=(0.30, 0.60),
            previous_window_s=(0.10, 0.30),
            next_window_s=None,
        )
        self.assertEqual(
            [(round(c.time_s, 2), c.attribution) for c in labelled],
            [
                (0.62, "target"),
                (0.40, "target"),
                (0.35, "target"),
                (0.29, "previous_word"),
                (0.70, "outside_target_window"),
            ],
        )
        self.assertAlmostEqual(labelled[0].distance_from_target_window_ms, 20.0)
        self.assertAlmostEqual(labelled[3].distance_from_target_window_ms, 10.0)
        with self.assertRaises(ValueError):
            attribute_release_candidates([], target_window_s=(0.5, 0.4))

    def test_two_pulses_separated_by_quiet_are_not_one_event(self) -> None:
        # Same-event grouping needs energy continuity, not just proximity.
        rng = np.random.default_rng(906)
        times = np.arange(int(self.SAMPLE_RATE * 0.8)) / self.SAMPLE_RATE
        waveform = rng.normal(0.0, 2e-5, len(times))
        for onset in (0.300, 0.335):
            pulse = (times >= onset) & (times < onset + 0.009)
            waveform[pulse] += rng.normal(0.0, 0.20, np.count_nonzero(pulse))
        estimate = self.estimate(waveform, 0.330, target_window_s=(0.25, 0.40))
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.release_time_s)
        self.assertIn("ambiguous_acoustic_top_candidate", estimate.exclusion_reasons)
        self.assertLess(float(estimate.runner_up_margin), 1.5)
        distinct = {c.time_s for c in estimate.candidates if not c.same_event_as_selected}
        self.assertTrue(distinct)

    def test_display_limit_cannot_hide_a_contest(self) -> None:
        waveform = synthetic_release_sequence(
            [(0.300, 0.20, 0.10, 0.345, 220.0), (0.420, 0.20, 0.10, 0.465, 260.0)]
        )
        for limit in (1, 8):
            with self.subTest(limit=limit):
                estimate = estimate_acoustic_release(
                    waveform,
                    self.SAMPLE_RATE,
                    window_start_s=0.0,
                    anchor_time_s=0.310,
                    config=AcousticReleaseConfig.conservative(maximum_candidates=limit),
                    target_window_s=(0.25, 0.385),
                    next_window_s=(0.385, 0.50),
                )
                self.assertFalse(estimate.measurable)
                self.assertIn("ambiguous_target_attribution", estimate.exclusion_reasons)
                self.assertLessEqual(len(estimate.candidates), limit)

    def test_overlapping_transcript_spans_own_nothing(self) -> None:
        waveform = synthetic_release_sequence([(0.350, 0.20, 0.10, None, 220.0)])
        estimate = self.estimate(
            waveform,
            0.340,
            target_window_s=(0.30, 0.45),
            next_window_s=(0.31, 0.50),
        )
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.candidate_time_s)
        self.assertEqual(estimate.exclusion_reasons, ("ambiguous_target_attribution",))
        self.assertTrue(
            all(c.attribution == "shared_word_span" for c in estimate.candidates)
        )

    def test_one_burst_is_not_measured_for_two_occurrences_in_one_word(self) -> None:
        waveform = synthetic_release_sequence([(0.350, 0.20, 0.10, None, 220.0)])
        anchors = (0.31, 0.37)
        estimates = [
            self.estimate(
                waveform,
                anchor,
                target_window_s=(0.28, 0.40),
                target_occurrence_index=index,
                target_occurrence_anchors_s=anchors,
            )
            for index, anchor in enumerate(anchors)
        ]
        for estimate in estimates:
            self.assertFalse(estimate.measurable)
            self.assertIsNone(estimate.candidate_time_s)
            self.assertEqual(
                estimate.exclusion_reasons,
                ("fewer_target_releases_than_bilabial_occurrences",),
            )
            self.assertEqual(estimate.target_occurrence_count, 2)

    def test_two_occurrences_are_assigned_in_transcript_order(self) -> None:
        waveform = synthetic_release_sequence(
            [(0.300, 0.20, 0.10, 0.345, 220.0), (0.420, 0.20, 0.10, 0.465, 260.0)]
        )
        anchors = (0.31, 0.43)
        first, second = [
            self.estimate(
                waveform,
                anchor,
                target_window_s=(0.28, 0.48),
                target_occurrence_index=index,
                target_occurrence_anchors_s=anchors,
            )
            for index, anchor in enumerate(anchors)
        ]
        self.assertTrue(first.measurable, first.exclusion_reasons)
        self.assertTrue(second.measurable, second.exclusion_reasons)
        self.assertLess(abs(float(first.release_time_s) - 0.300), 0.006)
        self.assertLess(abs(float(second.release_time_s) - 0.420), 0.006)
        # The other occurrence's release is not a rival for this one.
        self.assertIsNone(first.runner_up_margin)
        self.assertIsNone(second.runner_up_margin)

    def test_extra_in_word_release_resolves_only_when_anchors_disambiguate(self) -> None:
        # Three releases in one word (e.g. bilabial, velar, bilabial): the two
        # bilabial anchors sit near the first and last, so both resolve.
        waveform = synthetic_release_sequence(
            [
                (0.300, 0.20, 0.10, 0.345, 220.0),
                (0.400, 0.20, 0.10, 0.445, 240.0),
                (0.500, 0.20, 0.10, 0.545, 260.0),
            ]
        )
        anchors = (0.31, 0.51)
        first, second = [
            self.estimate(
                waveform,
                anchor,
                target_window_s=(0.28, 0.56),
                target_occurrence_index=index,
                target_occurrence_anchors_s=anchors,
            )
            for index, anchor in enumerate(anchors)
        ]
        self.assertLess(abs(float(first.candidate_time_s) - 0.300), 0.006)
        self.assertLess(abs(float(second.candidate_time_s) - 0.500), 0.006)
        # The unassigned middle release remains a runner-up for both.
        self.assertIsNotNone(first.runner_up_margin)
        self.assertIsNotNone(second.runner_up_margin)
        # Anchors that both point at the same release resolve nothing.
        collided = self.estimate(
            waveform,
            0.39,
            target_window_s=(0.28, 0.56),
            target_occurrence_index=0,
            target_occurrence_anchors_s=(0.39, 0.41),
        )
        self.assertFalse(collided.measurable)
        self.assertIn("unresolved_bilabial_occurrences_in_word", collided.exclusion_reasons)

    def test_invalid_windows_and_controls_raise(self) -> None:
        waveform = synthetic_release_sequence([(0.350, 0.20, 0.10, None, 220.0)])
        for window in ((-0.1, 0.5), (0.35, 0.35), (0.4, 0.3), (math.nan, 0.5)):
            for name in ("target_window_s", "previous_window_s", "next_window_s"):
                with self.subTest(name=name, window=window), self.assertRaises(ValueError):
                    self.estimate(
                        waveform, 0.34, **{"target_window_s": (0.3, 0.5), name: window}
                    )
        for override in (
            {"target_boundary_guard_ms": -1.0},
            {"same_event_window_ms": math.inf},
            {"same_event_continuity_db": math.nan},
            {"maximum_candidates": 0},
        ):
            with self.subTest(override=override), self.assertRaises((ValueError, TypeError)):
                estimate_acoustic_release(
                    waveform,
                    self.SAMPLE_RATE,
                    window_start_s=0.0,
                    anchor_time_s=0.34,
                    config=AcousticReleaseConfig.conservative(**override),
                )
        with self.assertRaises(ValueError):
            self.estimate(
                waveform,
                0.34,
                target_window_s=(0.3, 0.5),
                target_occurrence_index=2,
                target_occurrence_anchors_s=(0.32, 0.40),
            )

    def test_permissive_mode_also_respects_attribution(self) -> None:
        waveform = synthetic_release_sequence([(0.150, 0.30, 0.20, 0.32, 220.0)])
        estimate = estimate_acoustic_release(
            waveform,
            self.SAMPLE_RATE,
            window_start_s=0.0,
            anchor_time_s=0.380,
            target_window_s=self.TARGET,
            previous_window_s=self.PREVIOUS,
        )
        self.assertEqual(estimate.acceptance_mode, "permissive")
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.candidate_time_s)


class VisualReleaseTests(unittest.TestCase):
    def test_detects_confirmed_contact_and_reopening_at_native_pts(self) -> None:
        estimate = estimate_visual_release(
            aperture_sequence(), anchor_time_s=10.0 / 24.0
        )
        self.assertTrue(estimate.measurable)
        self.assertAlmostEqual(float(estimate.release_time_s), 10.0 / 24.0)
        self.assertEqual(len(estimate.closed_frame_times_s), 3)
        self.assertEqual(len(estimate.reopened_frame_times_s), 2)
        self.assertAlmostEqual(float(estimate.timing_uncertainty_ms), 1000.0 / 48.0)

    def test_long_near_duplicate_run_crossing_release_is_excluded(self) -> None:
        samples = aperture_sequence(
            closed_indices=range(5, 11), repeated_indices={7, 8, 9, 10}
        )
        estimate = estimate_visual_release(samples, anchor_time_s=11.0 / 24.0)
        self.assertFalse(estimate.measurable)
        self.assertIsNone(estimate.release_time_s)
        self.assertIn(
            "long_repeated_frame_run_crosses_release", estimate.exclusion_reasons
        )

    def test_low_resolution_face_is_excluded(self) -> None:
        samples = [
            LipApertureSample(
                **{**sample.as_dict(), "mouth_width_px": 60.0}
            )
            for sample in aperture_sequence()
        ]
        estimate = estimate_visual_release(samples, anchor_time_s=10.0 / 24.0)
        self.assertFalse(estimate.measurable)
        self.assertIn("insufficient_valid_face_frames", estimate.exclusion_reasons)

    def test_reports_raw_mouth_shape_at_acoustic_burst_frame(self) -> None:
        acoustic = estimate_acoustic_release(
            synthetic_plosive(),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.340,
        )
        shape = identify_mouth_shape_at_acoustic_release(
            aperture_sequence(), acoustic
        )
        self.assertTrue(shape.valid)
        self.assertEqual(shape.category, "closed/contact")
        self.assertEqual(len(shape.pair_apertures), 3)
        self.assertLess(abs(float(shape.frame_time_error_ms)), 25.0)


class CombinedMeasurementTests(unittest.TestCase):
    def test_positive_lag_means_visible_release_is_late(self) -> None:
        acoustic = estimate_acoustic_release(
            synthetic_plosive(),
            48_000,
            window_start_s=0.0,
            anchor_time_s=0.340,
        )
        visual = estimate_visual_release(
            aperture_sequence(), anchor_time_s=10.0 / 24.0
        )
        measurement = combine_release_estimates(
            acoustic, visual, aperture_sequence()
        )
        self.assertTrue(measurement.measurable)
        self.assertGreater(float(measurement.lag_ms), 0.0)
        self.assertEqual(
            measurement.mouth_shape_at_acoustic_release.category, "closed/contact"  # type: ignore[union-attr]
        )
        expected = (float(visual.release_time_s) - float(acoustic.release_time_s)) * 1000
        self.assertAlmostEqual(float(measurement.lag_ms), expected)


if __name__ == "__main__":
    unittest.main()
