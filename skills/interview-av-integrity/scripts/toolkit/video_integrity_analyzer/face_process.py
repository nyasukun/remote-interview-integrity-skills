from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from itertools import chain
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable

import numpy as np

from .media import VideoFrameSample
from .models import FaceTrack


class FaceWorkerError(RuntimeError):
    pass


_MACOS_SANDBOX_RUNTIME_MARKERS = (
    "DrishtiMetalHelper",
    "Service is unavailable",
)


def _worker_failure_message(exit_code: int | None, tail: str) -> str:
    """Return an actionable, fail-closed message for native worker failures."""

    base = f"face worker exited unexpectedly (code={exit_code})."
    if all(marker in tail for marker in _MACOS_SANDBOX_RUNTIME_MARKERS):
        base += (
            " MediaPipe's native macOS graph service could not start in this "
            "execution sandbox. VIDEO_INTEGRITY_FACE_TRANSPORT=file changes only "
            "how RGB frames reach the worker and cannot repair this Metal service "
            "failure. Re-run the face-runtime preflight and analyzer only in an "
            "approved local execution context outside the managed sandbox; do not "
            "treat this run as a measurement or a passing smoke test."
        )
    return f"{base} Log tail:\n{tail}"


def _read_message(process: subprocess.Popen[str], log_path: Path) -> dict[str, object]:
    if process.stdout is None:
        raise FaceWorkerError("face worker stdout is unavailable")
    line = process.stdout.readline()
    if line:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise FaceWorkerError(f"invalid face worker response: {line[:300]!r}") from error
        if message.get("status") == "error":
            raise FaceWorkerError(str(message.get("error", "face worker failed")))
        return message
    exit_code = process.poll()
    tail = ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        pass
    raise FaceWorkerError(_worker_failure_message(exit_code, tail))


def analyze_face_frames_isolated(
    frames: Iterable[VideoFrameSample],
    *,
    model_path: Path,
    max_faces: int,
    roi: tuple[float, float, float, float] | None,
    output_dir: Path,
    estimated_frames: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[FaceTrack], int, dict[str, int]]:
    """Run MediaPipe in a process that never imports PyAV.

    PyAV and MediaPipe's OpenCV wheel bundle different FFmpeg dylibs on macOS.
    Keeping them in separate processes avoids duplicate Objective-C class
    registration and turns a native detector abort into a normal Python error.
    RGB frames normally move through one fixed shared-memory buffer. Sandboxed
    environments that deny POSIX shared memory fall back to a file-backed
    memory map in the private temporary directory. Only timestamps and
    acknowledgements travel through pipes.
    """
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return [], 0, {"scene_cut_count": 0}
    shape = first.rgb.shape
    if len(shape) != 3 or shape[2] != 3 or first.rgb.dtype != np.uint8:
        raise ValueError("face frames must be uint8 RGB arrays")
    height, width, _ = shape
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "face_worker.log"

    process: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="video_integrity_face_") as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "tracks.pkl"
        cache_dir = temporary_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        shared: shared_memory.SharedMemory | None = None
        mapped: np.memmap | None = None
        transport_arguments: list[str]
        try:
            if os.environ.get("VIDEO_INTEGRITY_FACE_TRANSPORT") == "file":
                raise PermissionError("file-backed face transport requested")
            shared = shared_memory.SharedMemory(create=True, size=height * width * 3)
            shared_view = np.ndarray(shape, dtype=np.uint8, buffer=shared.buf)
            transport_arguments = ["--shared-memory", shared.name]
        except (PermissionError, OSError):
            frame_file = temporary_path / "frame-buffer.rgb"
            with frame_file.open("wb") as handle:
                handle.truncate(height * width * 3)
            mapped = np.memmap(frame_file, dtype=np.uint8, mode="r+", shape=shape)
            shared_view = mapped
            transport_arguments = ["--frame-file", str(frame_file)]
        command = [
            sys.executable,
            "-m",
            "video_integrity_analyzer.face_worker",
            *transport_arguments,
            "--height",
            str(height),
            "--width",
            str(width),
            "--model",
            str(model_path),
            "--max-faces",
            str(max_faces),
            "--result",
            str(result_path),
        ]
        if roi is not None:
            command.extend(("--roi", ",".join(str(value) for value in roi)))
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root
            if not existing_pythonpath
            else package_root + os.pathsep + existing_pythonpath
        )
        environment.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
        environment.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
        try:
            with log_path.open("w", encoding="utf-8") as worker_log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=worker_log,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
                ready = _read_message(process, log_path)
                if ready.get("status") != "ready":
                    raise FaceWorkerError(f"unexpected face worker startup response: {ready}")

                processed = 0
                for frame in chain((first,), iterator):
                    if frame.rgb.shape != shape:
                        raise FaceWorkerError(
                            f"video frame shape changed from {shape} to {frame.rgb.shape}"
                    )
                    np.copyto(shared_view, frame.rgb)
                    if mapped is not None:
                        mapped.flush()
                    if process.stdin is None:
                        raise FaceWorkerError("face worker stdin is unavailable")
                    process.stdin.write(
                        json.dumps({"command": "frame", "time_s": frame.time_s}) + "\n"
                    )
                    process.stdin.flush()
                    response = _read_message(process, log_path)
                    if response.get("status") != "frame":
                        raise FaceWorkerError(f"unexpected face worker response: {response}")
                    processed += 1
                    if progress is not None:
                        progress(processed, estimated_frames)
                if process.stdin is None:
                    raise FaceWorkerError("face worker stdin is unavailable")
                process.stdin.write('{"command":"finish"}\n')
                process.stdin.flush()
                response = _read_message(process, log_path)
                if response.get("status") != "finished":
                    raise FaceWorkerError(f"unexpected face worker finish response: {response}")
                process.stdin.close()
                exit_code = process.wait(timeout=60)
                if exit_code != 0:
                    tail = ""
                    try:
                        tail = log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )[-4000:]
                    except OSError:
                        pass
                    raise FaceWorkerError(_worker_failure_message(exit_code, tail))
            if not result_path.is_file():
                raise FaceWorkerError("face worker did not create its result file")
            with result_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                raise FaceWorkerError("face worker returned an invalid payload")
            tracks = payload.get("tracks")
            if not isinstance(tracks, list) or not all(isinstance(track, FaceTrack) for track in tracks):
                raise FaceWorkerError("face worker returned an invalid result object")
            return tracks, processed, {
                "scene_cut_count": int(payload.get("scene_cut_count", 0))
            }
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if mapped is not None:
                mapped.flush()
                del mapped
            if shared is not None:
                shared.close()
                try:
                    shared.unlink()
                except FileNotFoundError:
                    pass


def preflight_face_runtime(
    *,
    model_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Exercise the real isolated MediaPipe path with one synthetic frame.

    This catches native graph-service failures before a caller decodes or scans
    a long recording. The frame contains no case media and no person. A failed
    preflight is a hard runtime failure, never an unavailable observation.
    """

    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
    # A deterministic non-uniform image exercises image ingestion as well as
    # graph construction without looking like a face or containing case data.
    axis = np.arange(256, dtype=np.uint8)
    red = np.broadcast_to(axis[None, :], (256, 256))
    green = np.broadcast_to(axis[:, None], (256, 256))
    blue = np.bitwise_xor(red, green)
    rgb = np.stack((red, green, blue), axis=-1)
    frame = VideoFrameSample(
        time_s=0.0,
        rgb=rgb,
        source_pts=0,
        source_time_base="1/24",
    )
    tracks, processed, diagnostics = analyze_face_frames_isolated(
        (frame,),
        model_path=model_path,
        max_faces=1,
        roi=None,
        output_dir=output_dir,
        estimated_frames=1,
    )
    if processed != 1:
        raise FaceWorkerError(
            f"face-runtime preflight processed {processed} frames instead of 1"
        )
    return {
        "processed_frames": processed,
        "detected_tracks": len(tracks),
        "scene_cut_count": int(diagnostics.get("scene_cut_count", 0)),
    }
