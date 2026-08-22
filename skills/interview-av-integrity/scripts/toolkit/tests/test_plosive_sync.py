from __future__ import annotations

import math
import unittest

import numpy as np

from video_integrity_analyzer.plosive_sync import (
    AcousticReleaseConfig,
    LipApertureSample,
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
