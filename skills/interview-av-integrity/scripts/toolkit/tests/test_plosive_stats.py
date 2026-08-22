from __future__ import annotations

import unittest

from video_integrity_analyzer.plosive_stats import (
    compare_candidate_to_controls,
    summarize_epochs,
)


class PlosiveStatisticsTests(unittest.TestCase):
    def test_events_are_collapsed_before_comparison(self) -> None:
        events = [
            {"speaker": "c", "group": "candidate", "epoch_id": "c1", "release_lag_ms": 110.0},
            {"speaker": "c", "group": "candidate", "epoch_id": "c1", "release_lag_ms": 130.0},
            {"speaker": "c", "group": "candidate", "epoch_id": "c2", "release_lag_ms": 150.0},
            {"speaker": "i1", "group": "control", "epoch_id": "i1", "release_lag_ms": 10.0},
            {"speaker": "i2", "group": "control", "epoch_id": "i2", "release_lag_ms": 20.0},
            {
                "speaker": "i2",
                "group": "control",
                "epoch_id": "i2",
                "release_lag_ms": 900.0,
                "measurable": False,
            },
        ]
        epochs = summarize_epochs(events, margin_ms=83.34)
        self.assertEqual(len(epochs), 4)
        first = next(epoch for epoch in epochs if epoch.epoch_id == "c1")
        self.assertEqual(first.event_count, 2)
        self.assertEqual(first.median_lag_ms, 120.0)
        self.assertEqual(first.fraction_over_margin, 1.0)

        comparison = compare_candidate_to_controls(
            epochs,
            bootstrap_iterations=2_000,
            permutation_iterations=2_000,
        )
        self.assertEqual(comparison.candidate_epoch_count, 2)
        self.assertEqual(comparison.control_epoch_count, 2)
        self.assertGreater(float(comparison.median_difference_ms), 100.0)
        self.assertGreater(float(comparison.cliffs_delta), 0.9)

    def test_missing_group_returns_inconclusive_result(self) -> None:
        epochs = summarize_epochs(
            [
                {
                    "speaker": "c",
                    "group": "candidate",
                    "epoch_id": "c1",
                    "release_lag_ms": 100.0,
                }
            ]
        )
        result = compare_candidate_to_controls(epochs, bootstrap_iterations=10)
        self.assertIsNone(result.median_difference_ms)
        self.assertEqual(result.control_epoch_count, 0)


if __name__ == "__main__":
    unittest.main()
