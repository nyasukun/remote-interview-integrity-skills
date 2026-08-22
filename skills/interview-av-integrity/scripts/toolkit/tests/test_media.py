from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from video_integrity_analyzer.media import (
    decode_audio_features,
    iter_sampled_video_frames,
    probe_media,
    probe_video_frame_timing,
    save_rgb_thumbnail,
)


def make_test_media(path: Path) -> None:
    sample_rate = 48_000
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("mpeg4", rate=10)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"
        video.time_base = Fraction(1, 10)

        audio = container.add_stream("pcm_s16le", rate=sample_rate)
        audio.layout = "stereo"
        audio.time_base = Fraction(1, sample_rate)

        for index in range(20):
            rgb = np.zeros((48, 64, 3), dtype=np.uint8)
            rgb[:, :, 0] = index * 10
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        times = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
        amplitude = np.where((times >= 0.5) & (times < 1.4), 12_000.0, 50.0)
        waveform_left = (np.sin(2.0 * np.pi * 440.0 * times) * amplitude).astype(
            np.int16
        )
        waveform_right = (np.sin(2.0 * np.pi * 660.0 * times) * amplitude).astype(
            np.int16
        )
        for offset in range(0, len(waveform_left), 2048):
            frame = av.AudioFrame.from_ndarray(
                np.stack(
                    [
                        waveform_left[offset : offset + 2048],
                        waveform_right[offset : offset + 2048],
                    ]
                ),
                format="s16p",
                layout="stereo",
            )
            frame.sample_rate = sample_rate
            frame.pts = offset
            frame.time_base = Fraction(1, sample_rate)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


class MediaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.media_path = Path(self.temporary_directory.name) / "sample.mkv"
        make_test_media(self.media_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_probe_reports_primary_streams_and_optional_hash(self) -> None:
        probe = probe_media(self.media_path, compute_hash=True)
        self.assertEqual(probe.video_codec, "mpeg4")
        self.assertEqual((probe.width, probe.height), (64, 48))
        self.assertAlmostEqual(probe.average_fps, 10.0)
        self.assertEqual(probe.audio_codec, "pcm_s16le")
        self.assertEqual(probe.audio_sample_rate, 48_000)
        self.assertEqual(probe.audio_channels, 2)
        self.assertAlmostEqual(probe.duration_s, 2.0, places=2)
        self.assertEqual(len(probe.sha256 or ""), 64)

    def test_video_sampling_uses_real_pts(self) -> None:
        frames = list(
            iter_sampled_video_frames(
                self.media_path, start_s=0.25, end_s=1.25, fps=5.0
            )
        )
        times = [frame.time_s for frame in frames]
        self.assertEqual(len(frames), 5)
        np.testing.assert_allclose(times, [0.3, 0.5, 0.7, 0.9, 1.1], atol=1e-6)
        self.assertEqual(frames[0].source_pts, 300)
        self.assertEqual(frames[0].rgb.shape, (48, 64, 3))
        unpacked_time, unpacked_rgb = frames[0]
        self.assertEqual(unpacked_time, frames[0].time_s)
        self.assertIs(unpacked_rgb, frames[0].rgb)

    def test_complete_pts_scan_distinguishes_expected_cfr(self) -> None:
        matched = probe_video_frame_timing(self.media_path, expected_fps=10.0)
        self.assertTrue(matched.is_cfr)
        self.assertEqual(matched.decoded_frame_count, 20)
        mismatched = probe_video_frame_timing(self.media_path, expected_fps=24.0)
        self.assertFalse(mismatched.is_cfr)
        self.assertGreater(mismatched.outlier_interval_count, 0)

    def test_audio_features_preserve_grid_and_level_change(self) -> None:
        features = decode_audio_features(
            self.media_path, start_s=0.2, end_s=1.8, rate_hz=100.0
        )
        self.assertEqual(len(features.grid_s), 160)
        self.assertAlmostEqual(features.grid_s[0], 0.205)
        self.assertAlmostEqual(features.grid_s[-1], 1.795)
        self.assertGreater(features.diagnostics.coverage_fraction, 0.98)
        self.assertEqual(features.diagnostics.discontinuity_gaps, 0)
        self.assertEqual(features.diagnostics.overlapping_chunks, 0)

        loud = (features.grid_s >= 0.7) & (features.grid_s <= 1.2)
        quiet = (features.grid_s >= 0.25) & (features.grid_s <= 0.4)
        self.assertGreater(
            float(np.nanmedian(features.envelope[loud])),
            float(np.nanmedian(features.envelope[quiet])) + 20.0,
        )
        self.assertGreater(np.count_nonzero(features.speech_mask[loud]), 40)
        self.assertEqual(np.count_nonzero(features.speech_mask[quiet]), 0)
        self.assertGreater(np.count_nonzero(np.isfinite(features.transition)), 150)

    def test_thumbnail_is_resized_without_aspect_distortion(self) -> None:
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        output = Path(self.temporary_directory.name) / "thumb.jpg"
        saved = save_rgb_thumbnail(rgb, output, max_width=80)
        with Image.open(saved) as image:
            self.assertEqual(image.size, (80, 40))


if __name__ == "__main__":
    unittest.main()
