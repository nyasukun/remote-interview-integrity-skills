from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video_integrity_analyzer import face_process


class FaceProcessTests(unittest.TestCase):
    def test_macos_native_service_abort_has_actionable_fail_closed_message(self) -> None:
        message = face_process._worker_failure_message(
            -6,
            "Check failed: service_ Service is unavailable.\n"
            "-[DrishtiMetalHelper initWithCalculatorContext:]",
        )
        self.assertIn("native macOS graph service", message)
        self.assertIn("VIDEO_INTEGRITY_FACE_TRANSPORT=file", message)
        self.assertIn("approved local execution context", message)
        self.assertIn("do not treat this run as a measurement", message)

    def test_unrelated_worker_failure_is_not_misclassified_as_sandbox_issue(self) -> None:
        message = face_process._worker_failure_message(1, "ordinary Python traceback")
        self.assertIn("code=1", message)
        self.assertNotIn("native macOS graph service", message)

    def test_preflight_uses_one_synthetic_frame_and_requires_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "face.task"
            model.write_bytes(b"test model placeholder")
            with patch.object(
                face_process,
                "analyze_face_frames_isolated",
                return_value=([], 1, {"scene_cut_count": 0}),
            ) as isolated:
                result = face_process.preflight_face_runtime(
                    model_path=model,
                    output_dir=root / "preflight",
                )

            frames = isolated.call_args.args[0]
            frame = tuple(frames)[0]
            self.assertEqual(frame.rgb.shape, (256, 256, 3))
            self.assertEqual(frame.rgb.dtype, np.uint8)
            self.assertGreater(int(np.ptp(frame.rgb)), 0)
            self.assertEqual(result["processed_frames"], 1)
            self.assertEqual(result["detected_tracks"], 0)


if __name__ == "__main__":
    unittest.main()
