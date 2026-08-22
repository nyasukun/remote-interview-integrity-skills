from __future__ import annotations

import unittest

from video_integrity_analyzer.face import FaceTracker, _roi_pixels
from video_integrity_analyzer.models import FaceDetection


def detection(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    appearance: tuple[float, ...] = (),
) -> FaceDetection:
    return FaceDetection(
        bbox=(x0, y0, x1, y1),
        center=((x0 + x1) * 0.5, (y0 + y1) * 0.5),
        face_width_px=160.0,
        mouth_width_px=40.0,
        mouth_open=0.1,
        mouth_texture_ratio=1.0,
        quality=1.0,
        normalized_lip_shape=(0.0, 0.0),
        appearance_descriptor=appearance,
    )


class FaceTrackerTests(unittest.TestCase):
    def test_keeps_tracks_when_detector_order_changes(self) -> None:
        tracker = FaceTracker()
        first = tracker.update(0.0, [detection(0.10, 0.10, 0.30, 0.40), detection(0.65, 0.10, 0.85, 0.40)])
        self.assertEqual([track_id for track_id, _ in first], [0, 1])

        second = tracker.update(0.1, [detection(0.64, 0.10, 0.84, 0.40), detection(0.11, 0.10, 0.31, 0.40)])
        self.assertEqual([track_id for track_id, _ in second], [0, 1])
        self.assertEqual([len(track.samples) for track in tracker.tracks], [2, 2])
        self.assertLess(tracker.tracks[0].last_bbox[0], tracker.tracks[1].last_bbox[0])

    def test_new_track_after_long_gap(self) -> None:
        tracker = FaceTracker(max_gap_s=0.5)
        tracker.update(0.0, [detection(0.1, 0.1, 0.3, 0.4)])
        assignment = tracker.update(1.0, [detection(0.1, 0.1, 0.3, 0.4)])
        self.assertEqual(assignment[0][0], 1)

    def test_active_speaker_cut_does_not_merge_people_at_same_bbox(self) -> None:
        tracker = FaceTracker()
        bbox = (0.1, 0.1, 0.9, 0.9)
        tracker.update(0.0, [detection(*bbox, appearance=(1.0, 0.0, 0.0))])
        switched = tracker.update(0.1, [detection(*bbox, appearance=(0.0, 1.0, 0.0))])
        self.assertEqual(switched[0][0], 1)

    def test_returning_speaker_reconnects_by_appearance(self) -> None:
        tracker = FaceTracker(max_gap_s=0.5, max_reconnect_gap_s=3600.0)
        bbox = (0.1, 0.1, 0.9, 0.9)
        tracker.update(0.0, [detection(*bbox, appearance=(1.0, 0.0, 0.0))])
        tracker.update(1.0, [detection(*bbox, appearance=(0.0, 1.0, 0.0))])
        returned = tracker.update(2.0, [detection(*bbox, appearance=(0.995, 0.005, 0.0))])
        self.assertEqual(returned[0][0], 0)
        self.assertEqual(len(tracker.tracks), 2)

    def test_explicit_scene_cut_forces_new_segment(self) -> None:
        tracker = FaceTracker()
        bbox = (0.1, 0.1, 0.9, 0.9)
        descriptor = (1.0, 0.0, 0.0)
        tracker.update(0.0, [detection(*bbox, appearance=descriptor)])
        tracker.start_new_scene()
        assignment = tracker.update(0.1, [detection(*bbox, appearance=descriptor)])
        self.assertEqual(assignment[0][0], 1)

    def test_roi_rounding_stays_inside_frame(self) -> None:
        self.assertEqual(_roi_pixels((0.1, 0.2, 0.9, 0.8), 100, 50), (10, 10, 90, 40))
        with self.assertRaises(ValueError):
            _roi_pixels((0.9, 0.1, 0.2, 0.8), 100, 50)


if __name__ == "__main__":
    unittest.main()
