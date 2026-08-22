from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np
from PIL import Image

from .models import MediaProbe


PathLike = str | os.PathLike[str]


class MediaError(RuntimeError):
    """Base class for errors that prevent media analysis."""


class NoVideoStreamError(MediaError):
    """Raised when an operation requires video but the file has no video stream."""


class NoAudioStreamError(MediaError):
    """Raised when an operation requires audio but the file has no audio stream."""


@dataclass(frozen=True)
class VideoFrameSample:
    """A decoded RGB frame and its real presentation timestamp.

    ``time_s`` is normalized to the container timeline while preserving the
    offset between audio and video streams.  The object can also be unpacked
    as ``time_s, rgb`` for callers that do not need the source PTS metadata.
    """

    time_s: float
    rgb: np.ndarray
    source_pts: int
    source_time_base: str

    def __iter__(self) -> Iterator[Any]:
        yield self.time_s
        yield self.rgb


@dataclass(frozen=True)
class AudioDiagnostics:
    stream_index: int
    source_sample_rate: int | None
    target_sample_rate: int
    feature_rate_hz: float
    start_s: float
    end_s: float | None
    decoded_frames: int
    resampled_frames: int
    input_frames_without_pts: int
    output_frames_without_pts: int
    pts_reversals: int
    discontinuity_gaps: int
    overlapping_chunks: int
    total_bins: int
    valid_bins: int
    coverage_fraction: float
    noise_floor_dbfs: float | None
    speech_threshold_dbfs: float | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioFeatures:
    """Audio features on a timestamped, uniformly spaced analysis grid.

    The grid values are centers of adjacent feature bins. ``envelope`` is a
    25 ms-equivalent log-RMS envelope in dBFS. ``transition`` combines changes
    in level and waveform derivative energy. Missing or discontinuous audio is
    represented by NaN rather than silently bridged. ``speech_mask`` is an
    energy-based aid for selecting useful windows, not speaker recognition or
    a semantic speech detector.
    """

    grid_s: np.ndarray
    envelope: np.ndarray
    transition: np.ndarray
    speech_mask: np.ndarray
    activity: np.ndarray
    diagnostics: AudioDiagnostics


@dataclass(frozen=True)
class FrameTimingDiagnostics:
    expected_fps: float
    expected_interval_s: float
    decoded_frame_count: int
    interval_count: int
    missing_pts_count: int
    nonpositive_interval_count: int
    outlier_interval_count: int
    outlier_fraction: float
    median_interval_s: float | None
    maximum_absolute_deviation_s: float | None
    tolerance_s: float
    is_cfr: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _checked_path(path: PathLike) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _stream_start_raw_s(stream: av.stream.Stream) -> float | None:
    if stream.start_time is None or stream.time_base is None:
        return None
    return float(stream.start_time * stream.time_base)


def _timeline_origin_s(container: av.container.InputContainer) -> float:
    """Return one common origin so cross-stream offsets remain intact."""
    if container.start_time is not None:
        return float(container.start_time / av.time_base)
    starts = [
        start
        for stream in container.streams
        if (start := _stream_start_raw_s(stream)) is not None
    ]
    return min(starts) if starts else 0.0


def _duration_s(container: av.container.InputContainer, origin_s: float) -> float:
    if container.duration is not None:
        return max(0.0, float(container.duration / av.time_base))
    ends: list[float] = []
    for stream in container.streams:
        if stream.duration is None or stream.time_base is None:
            continue
        start = _stream_start_raw_s(stream)
        raw_end = (start if start is not None else origin_s) + float(
            stream.duration * stream.time_base
        )
        ends.append(raw_end - origin_s)
    return max(ends, default=0.0)


def _codec_name(stream: av.stream.Stream) -> str:
    context = stream.codec_context
    return str(getattr(context, "name", None) or getattr(context.codec, "name", "unknown"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_media(path: PathLike, compute_hash: bool = False) -> MediaProbe:
    """Read container/stream metadata without decoding the media payload."""
    source = _checked_path(path)
    with av.open(str(source), mode="r") as container:
        video = next(iter(container.streams.video), None)
        if video is None:
            raise NoVideoStreamError(f"No video stream found in {source}")
        audio = next(iter(container.streams.audio), None)
        origin_s = _timeline_origin_s(container)

        video_start_raw = _stream_start_raw_s(video)
        video_start_s = (video_start_raw - origin_s) if video_start_raw is not None else 0.0
        average_rate = video.average_rate or video.base_rate or video.guessed_rate
        average_fps = float(average_rate) if average_rate else 0.0

        audio_codec: str | None = None
        audio_sample_rate: int | None = None
        audio_channels: int | None = None
        audio_time_base: str | None = None
        audio_start_s: float | None = None
        if audio is not None:
            audio_codec = _codec_name(audio)
            sample_rate_value = getattr(audio.codec_context, "sample_rate", None)
            audio_sample_rate = int(sample_rate_value) if sample_rate_value else None
            layout = getattr(audio.codec_context, "layout", None)
            channel_value = getattr(layout, "nb_channels", None)
            if channel_value is None:
                channel_value = getattr(audio.codec_context, "channels", None)
            audio_channels = int(channel_value) if channel_value else None
            audio_time_base = str(audio.time_base) if audio.time_base is not None else None
            audio_start_raw = _stream_start_raw_s(audio)
            audio_start_s = (
                audio_start_raw - origin_s if audio_start_raw is not None else 0.0
            )

        result = MediaProbe(
            path=str(source),
            size_bytes=source.stat().st_size,
            duration_s=_duration_s(container, origin_s),
            video_codec=_codec_name(video),
            width=int(video.codec_context.width),
            height=int(video.codec_context.height),
            average_fps=average_fps,
            video_time_base=str(video.time_base),
            video_start_s=video_start_s,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
            audio_time_base=audio_time_base,
            audio_start_s=audio_start_s,
            container_format=str(container.format.name),
            sha256=None,
        )
    if compute_hash:
        # Hashing is deliberately outside the open demuxer so the sequential
        # read cannot disturb decoder state in callers that reuse this result.
        result = MediaProbe(**{**asdict(result), "sha256": _sha256(source)})
    return result


def probe_video_frame_timing(
    path: PathLike,
    *,
    expected_fps: float,
    relative_tolerance: float = 0.02,
) -> FrameTimingDiagnostics:
    """Decode presentation timestamps and verify a constant frame cadence.

    This scans the complete video stream but does not convert frames to RGB.
    A single missing, nonpositive, or cadence-outlier interval fails the strict
    CFR result used by the evidence workflow.
    """

    if not math.isfinite(expected_fps) or expected_fps <= 0:
        raise ValueError("expected_fps must be finite and positive")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 0.5:
        raise ValueError("relative_tolerance must be between 0 and 0.5")
    source = _checked_path(path)
    expected_interval = 1.0 / expected_fps
    tolerance = max(1e-6, expected_interval * relative_tolerance)
    timestamps: list[float] = []
    missing_pts = 0
    with av.open(str(source), mode="r") as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise NoVideoStreamError(f"No video stream found in {source}")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                missing_pts += 1
                continue
            timestamps.append(float(frame.pts * frame.time_base))
    if len(timestamps) < 2:
        return FrameTimingDiagnostics(
            expected_fps=expected_fps,
            expected_interval_s=expected_interval,
            decoded_frame_count=len(timestamps) + missing_pts,
            interval_count=max(0, len(timestamps) - 1),
            missing_pts_count=missing_pts,
            nonpositive_interval_count=0,
            outlier_interval_count=0,
            outlier_fraction=1.0,
            median_interval_s=None,
            maximum_absolute_deviation_s=None,
            tolerance_s=tolerance,
            is_cfr=False,
        )
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    deviations = np.abs(intervals - expected_interval)
    nonpositive = int(np.count_nonzero(intervals <= 0.0))
    outliers = int(np.count_nonzero(deviations > tolerance))
    return FrameTimingDiagnostics(
        expected_fps=expected_fps,
        expected_interval_s=expected_interval,
        decoded_frame_count=len(timestamps) + missing_pts,
        interval_count=int(intervals.size),
        missing_pts_count=missing_pts,
        nonpositive_interval_count=nonpositive,
        outlier_interval_count=outliers,
        outlier_fraction=float(outliers / intervals.size),
        median_interval_s=float(np.median(intervals)),
        maximum_absolute_deviation_s=float(np.max(deviations)),
        tolerance_s=tolerance,
        is_cfr=missing_pts == 0 and nonpositive == 0 and outliers == 0,
    )


def _seek(
    container: av.container.InputContainer,
    stream: av.stream.Stream,
    normalized_time_s: float,
    origin_s: float,
) -> str | None:
    """Seek to a preceding keyframe; return a warning instead of hiding failure."""
    raw_time_s = max(0.0, normalized_time_s + origin_s)
    try:
        if stream.time_base is not None:
            offset = int(math.floor(raw_time_s / float(stream.time_base)))
            container.seek(offset, stream=stream, backward=True, any_frame=False)
        else:
            container.seek(
                int(math.floor(raw_time_s * av.time_base)), backward=True, any_frame=False
            )
        return None
    except Exception as error:  # PyAV exposes format-specific FFmpeg exception subclasses.
        return f"seek failed; decoded from stream start ({type(error).__name__}: {error})"


def _rotation_degrees(stream: av.video.stream.VideoStream) -> int:
    value = stream.metadata.get("rotate")
    if value is None:
        return 0
    try:
        return int(round(float(value) / 90.0) * 90) % 360
    except (TypeError, ValueError):
        return 0


def _apply_rotation(rgb: np.ndarray, rotation_degrees: int) -> np.ndarray:
    if rotation_degrees == 90:
        return np.ascontiguousarray(np.rot90(rgb, k=3))
    if rotation_degrees == 180:
        return np.ascontiguousarray(np.rot90(rgb, k=2))
    if rotation_degrees == 270:
        return np.ascontiguousarray(np.rot90(rgb, k=1))
    return rgb


def iter_sampled_video_frames(
    path: PathLike,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    fps: float = 10.0,
) -> Iterator[VideoFrameSample]:
    """Yield sampled RGB video frames using decoded presentation timestamps.

    The function never fabricates timestamps from frame count or nominal FPS.
    Frames with missing, duplicate, or reversed PTS are skipped. Sampling emits
    at most one real decoded frame per ``1 / fps`` interval, and each emitted
    sample retains the source PTS.
    """
    source = _checked_path(path)
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be a finite non-negative value")
    if end_s is not None and (not math.isfinite(end_s) or end_s <= start_s):
        raise ValueError("end_s must be finite and greater than start_s")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive value")

    with av.open(str(source), mode="r") as container:
        video = next(iter(container.streams.video), None)
        if video is None:
            raise NoVideoStreamError(f"No video stream found in {source}")
        origin_s = _timeline_origin_s(container)
        # A two-second preroll is enough for the common long-GOP interview
        # formats while still letting the decoder find the actual keyframe.
        _seek(container, video, max(0.0, start_s - 2.0), origin_s)
        rotation = _rotation_degrees(video)
        interval_s = 1.0 / fps
        next_due_s = start_s
        last_source_time_s = -math.inf
        finished = False

        for packet in container.demux(video):
            for frame in packet.decode():
                if frame.pts is None or frame.time_base is None:
                    continue
                raw_time_s = float(frame.pts * frame.time_base)
                time_s = raw_time_s - origin_s
                if time_s <= last_source_time_s + 1e-9:
                    continue
                last_source_time_s = time_s
                if time_s + 1e-9 < start_s:
                    continue
                if end_s is not None and time_s >= end_s:
                    finished = True
                    break
                if time_s + 1e-9 < next_due_s:
                    continue

                rgb = frame.to_ndarray(format="rgb24")
                rgb = _apply_rotation(rgb, rotation)
                yield VideoFrameSample(
                    time_s=time_s,
                    rgb=rgb,
                    source_pts=int(frame.pts),
                    source_time_base=str(frame.time_base),
                )
                # Advance past every nominal sampling instant crossed by this
                # real frame, without duplicating a frame after a PTS gap.
                crossed = max(1, int(math.floor((time_s - next_due_s) / interval_s)) + 1)
                next_due_s += crossed * interval_s
            if finished:
                break


def _window_overlap_kernel(rate_hz: float, window_s: float) -> np.ndarray:
    """Weights feature bins by their exact overlap with a centered window."""
    hop_s = 1.0 / rate_hz
    half_window = window_s / 2.0
    maximum_offset = int(math.ceil((half_window + hop_s / 2.0) / hop_s))
    weights: list[float] = []
    for offset in range(-maximum_offset, maximum_offset + 1):
        bin_left = offset * hop_s - hop_s / 2.0
        bin_right = offset * hop_s + hop_s / 2.0
        overlap = max(0.0, min(bin_right, half_window) - max(bin_left, -half_window))
        weights.append(overlap / hop_s)
    kernel = np.asarray(weights, dtype=np.float64)
    return kernel[kernel > 1e-12]


def _nan_centered_smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64)
    finite = np.isfinite(values)
    sums = _convolve_centered(np.where(finite, values, 0.0), kernel)
    counts = _convolve_centered(finite.astype(np.float64), kernel)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts >= max(1.0, width * 0.6))
    return result


def _convolve_centered(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return centered convolution with exactly ``len(values)`` outputs."""
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    full = np.convolve(values, kernel, mode="full")
    start = (len(kernel) - 1) // 2
    return full[start : start + len(values)]


def _close_short_boolean_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = mask.copy()
    if maximum_gap <= 0:
        return result
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        stop = index
        while stop < len(result) and not result[stop]:
            stop += 1
        if index > 0 and stop < len(result) and stop - index <= maximum_gap:
            result[index:stop] = True
        index = stop
    return result


def _remove_short_boolean_runs(mask: np.ndarray, minimum_run: int) -> np.ndarray:
    result = mask.copy()
    if minimum_run <= 1:
        return result
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        stop = index
        while stop < len(result) and result[stop]:
            stop += 1
        if stop - index < minimum_run:
            result[index:stop] = False
        index = stop
    return result


def _empty_audio_features(
    *,
    stream_index: int,
    source_sample_rate: int | None,
    sample_rate: int,
    rate_hz: float,
    start_s: float,
    end_s: float | None,
    warnings: tuple[str, ...],
) -> AudioFeatures:
    empty_float = np.empty(0, dtype=np.float64)
    diagnostics = AudioDiagnostics(
        stream_index=stream_index,
        source_sample_rate=source_sample_rate,
        target_sample_rate=sample_rate,
        feature_rate_hz=rate_hz,
        start_s=start_s,
        end_s=end_s,
        decoded_frames=0,
        resampled_frames=0,
        input_frames_without_pts=0,
        output_frames_without_pts=0,
        pts_reversals=0,
        discontinuity_gaps=0,
        overlapping_chunks=0,
        total_bins=0,
        valid_bins=0,
        coverage_fraction=0.0,
        noise_floor_dbfs=None,
        speech_threshold_dbfs=None,
        warnings=warnings,
    )
    return AudioFeatures(
        grid_s=empty_float.copy(),
        envelope=empty_float.copy(),
        transition=empty_float.copy(),
        speech_mask=np.empty(0, dtype=bool),
        activity=empty_float.copy(),
        diagnostics=diagnostics,
    )


def decode_audio_features(
    path: PathLike,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    rate_hz: float = 100.0,
    sample_rate: int = 16_000,
) -> AudioFeatures:
    """Decode audio and calculate PTS-aware features without storing full PCM.

    Audio is resampled to mono 16-bit PCM, then accumulated directly into
    timestamped feature bins. This keeps memory proportional to duration times
    ``rate_hz`` instead of duration times PCM sample rate. Gaps and overlaps are
    detected from resampled frame PTS and are disclosed in diagnostics.
    """
    source = _checked_path(path)
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be a finite non-negative value")
    if end_s is not None and (not math.isfinite(end_s) or end_s <= start_s):
        raise ValueError("end_s must be finite and greater than start_s")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be a finite positive value")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples_per_bin = sample_rate / rate_hz
    if abs(samples_per_bin - round(samples_per_bin)) > 1e-9:
        raise ValueError("sample_rate / rate_hz must be an integer")

    with av.open(str(source), mode="r") as container:
        audio = next(iter(container.streams.audio), None)
        if audio is None:
            raise NoAudioStreamError(f"No audio stream found in {source}")
        origin_s = _timeline_origin_s(container)
        duration_s = _duration_s(container, origin_s)
        effective_end_s = end_s
        if duration_s > 0:
            effective_end_s = (
                duration_s if effective_end_s is None else min(effective_end_s, duration_s)
            )
        if effective_end_s is not None and effective_end_s <= start_s:
            return _empty_audio_features(
                stream_index=int(audio.index),
                source_sample_rate=getattr(audio.codec_context, "sample_rate", None),
                sample_rate=sample_rate,
                rate_hz=rate_hz,
                start_s=start_s,
                end_s=effective_end_s,
                warnings=("requested range lies outside the media duration",),
            )

        warnings: list[str] = []
        seek_warning = _seek(container, audio, max(0.0, start_s - 0.1), origin_s)
        if seek_warning:
            warnings.append(seek_warning)

        initial_bins = (
            int(math.ceil((effective_end_s - start_s) * rate_hz - 1e-9))
            if effective_end_s is not None
            else 0
        )
        energy_sum = np.zeros(initial_bins, dtype=np.float64)
        derivative_sum = np.zeros(initial_bins, dtype=np.float64)
        sample_count = np.zeros(initial_bins, dtype=np.int64)
        derivative_count = np.zeros(initial_bins, dtype=np.int64)
        maximum_touched_bin = -1

        def ensure_size(size: int) -> None:
            nonlocal energy_sum, derivative_sum, sample_count, derivative_count
            if size <= len(energy_sum):
                return
            growth = max(size, max(1024, len(energy_sum) * 2))
            energy_sum = np.pad(energy_sum, (0, growth - len(energy_sum)))
            derivative_sum = np.pad(derivative_sum, (0, growth - len(derivative_sum)))
            sample_count = np.pad(sample_count, (0, growth - len(sample_count)))
            derivative_count = np.pad(
                derivative_count, (0, growth - len(derivative_count))
            )

        decoded_frames = 0
        resampled_frames = 0
        input_without_pts = 0
        output_without_pts = 0
        pts_reversals = 0
        discontinuity_gaps = 0
        overlapping_chunks = 0
        last_chunk_start_s: float | None = None
        last_chunk_end_s: float | None = None
        previous_sample: float | None = None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        # Some containers expose audio PTS only to the nearest millisecond.
        # Treat sub-time-base jitter as timestamp quantization, not as a real
        # discontinuity that should cause samples to be removed or duplicated.
        source_time_base_s = float(audio.time_base) if audio.time_base is not None else 0.0
        timestamp_tolerance_s = max(2.0 / sample_rate, 1.5 * source_time_base_s)

        def accumulate(frame: av.AudioFrame) -> None:
            nonlocal maximum_touched_bin, resampled_frames, output_without_pts
            nonlocal pts_reversals, discontinuity_gaps, overlapping_chunks
            nonlocal last_chunk_start_s, last_chunk_end_s, previous_sample
            resampled_frames += 1
            array = np.asarray(frame.to_ndarray()).reshape(-1)
            if array.size == 0:
                return
            waveform = array.astype(np.float64) / 32768.0
            if frame.pts is not None and frame.time_base is not None:
                chunk_start_s = float(frame.pts * frame.time_base) - origin_s
            elif last_chunk_end_s is not None:
                output_without_pts += 1
                chunk_start_s = last_chunk_end_s
            else:
                output_without_pts += 1
                warnings.append("first resampled audio frame had no PTS and was skipped")
                return

            if last_chunk_start_s is not None and chunk_start_s < last_chunk_start_s - 1e-6:
                pts_reversals += 1
            original_chunk_end_s = chunk_start_s + len(waveform) / sample_rate
            contiguous = False
            if last_chunk_end_s is not None:
                delta_s = chunk_start_s - last_chunk_end_s
                contiguous = abs(delta_s) <= timestamp_tolerance_s
                if delta_s > timestamp_tolerance_s:
                    discontinuity_gaps += 1
                    previous_sample = None
                elif delta_s < -timestamp_tolerance_s:
                    overlapping_chunks += 1
                    skip = min(
                        len(waveform),
                        max(0, int(math.ceil(-delta_s * sample_rate - 1e-6))),
                    )
                    if skip:
                        waveform = waveform[skip:]
                        chunk_start_s += skip / sample_rate
                    contiguous = (
                        abs(chunk_start_s - last_chunk_end_s) <= timestamp_tolerance_s
                    )
            if waveform.size == 0:
                last_chunk_start_s = chunk_start_s
                last_chunk_end_s = max(last_chunk_end_s or -math.inf, original_chunk_end_s)
                return

            differences = np.empty(waveform.shape, dtype=np.float64)
            if previous_sample is not None and contiguous:
                differences[0] = waveform[0] - previous_sample
            else:
                differences[0] = np.nan
            differences[1:] = np.diff(waveform)

            sample_times_s = chunk_start_s + np.arange(len(waveform)) / sample_rate
            in_range = sample_times_s >= start_s - 1e-10
            if effective_end_s is not None:
                in_range &= sample_times_s < effective_end_s - 1e-10
            if np.any(in_range):
                selected_times = sample_times_s[in_range]
                selected_waveform = waveform[in_range]
                selected_differences = differences[in_range]
                bins = np.floor((selected_times - start_s) * rate_hz + 1e-8).astype(
                    np.int64
                )
                nonnegative = bins >= 0
                bins = bins[nonnegative]
                selected_waveform = selected_waveform[nonnegative]
                selected_differences = selected_differences[nonnegative]
                if bins.size:
                    first_bin = int(bins.min())
                    last_bin = int(bins.max())
                    ensure_size(last_bin + 1)
                    maximum_touched_bin = max(maximum_touched_bin, last_bin)
                    local_bins = bins - first_bin
                    bin_count = last_bin - first_bin + 1
                    energy_sum[first_bin : last_bin + 1] += np.bincount(
                        local_bins, weights=selected_waveform**2, minlength=bin_count
                    )
                    sample_count[first_bin : last_bin + 1] += np.bincount(
                        local_bins, minlength=bin_count
                    )
                    finite_difference = np.isfinite(selected_differences)
                    if np.any(finite_difference):
                        derivative_bins = bins[finite_difference]
                        derivative_values = selected_differences[finite_difference]
                        first_derivative_bin = int(derivative_bins.min())
                        last_derivative_bin = int(derivative_bins.max())
                        local_derivative_bins = derivative_bins - first_derivative_bin
                        derivative_bin_count = (
                            last_derivative_bin - first_derivative_bin + 1
                        )
                        derivative_sum[
                            first_derivative_bin : last_derivative_bin + 1
                        ] += np.bincount(
                            local_derivative_bins,
                            weights=derivative_values**2,
                            minlength=derivative_bin_count,
                        )
                        derivative_count[
                            first_derivative_bin : last_derivative_bin + 1
                        ] += np.bincount(
                            local_derivative_bins, minlength=derivative_bin_count
                        )

            previous_sample = float(waveform[-1])
            last_chunk_start_s = chunk_start_s
            last_chunk_end_s = max(last_chunk_end_s or -math.inf, original_chunk_end_s)

        stopped_for_end = False
        for packet in container.demux(audio):
            for frame in packet.decode():
                decoded_frames += 1
                if frame.pts is None or frame.time_base is None:
                    input_without_pts += 1
                    input_time_s = None
                else:
                    input_time_s = float(frame.pts * frame.time_base) - origin_s
                if effective_end_s is not None and input_time_s is not None:
                    if input_time_s > effective_end_s + 0.1:
                        stopped_for_end = True
                        break
                for output in resampler.resample(frame):
                    accumulate(output)
            if stopped_for_end:
                break
        for output in resampler.resample(None):
            accumulate(output)

        if effective_end_s is not None:
            number_of_bins = initial_bins
        else:
            number_of_bins = maximum_touched_bin + 1
            effective_end_s = (
                start_s + number_of_bins / rate_hz if number_of_bins > 0 else None
            )
        energy_sum = energy_sum[:number_of_bins]
        derivative_sum = derivative_sum[:number_of_bins]
        sample_count = sample_count[:number_of_bins]
        derivative_count = derivative_count[:number_of_bins]

        if number_of_bins == 0:
            return _empty_audio_features(
                stream_index=int(audio.index),
                source_sample_rate=getattr(audio.codec_context, "sample_rate", None),
                sample_rate=sample_rate,
                rate_hz=rate_hz,
                start_s=start_s,
                end_s=effective_end_s,
                warnings=tuple(warnings or ["audio decoder returned no samples"]),
            )

        kernel = _window_overlap_kernel(rate_hz, 0.025)
        weighted_energy = _convolve_centered(energy_sum, kernel)
        weighted_count = _convolve_centered(sample_count.astype(np.float64), kernel)
        weighted_derivative = _convolve_centered(derivative_sum, kernel)
        weighted_derivative_count = _convolve_centered(
            derivative_count.astype(np.float64), kernel
        )
        minimum_window_samples = sample_rate * 0.025 * 0.65
        valid = weighted_count >= minimum_window_samples
        derivative_valid = weighted_derivative_count >= minimum_window_samples * 0.65

        mean_square = np.full(number_of_bins, np.nan, dtype=np.float64)
        np.divide(weighted_energy, weighted_count, out=mean_square, where=valid)
        envelope = np.full(number_of_bins, np.nan, dtype=np.float64)
        envelope[valid] = 10.0 * np.log10(np.maximum(mean_square[valid], 1e-14))

        derivative_mean_square = np.full(number_of_bins, np.nan, dtype=np.float64)
        np.divide(
            weighted_derivative,
            weighted_derivative_count,
            out=derivative_mean_square,
            where=derivative_valid,
        )
        activity = np.full(number_of_bins, np.nan, dtype=np.float64)
        activity[derivative_valid] = 10.0 * np.log10(
            np.maximum(derivative_mean_square[derivative_valid], 1e-14)
        )

        level_change = np.full(number_of_bins, np.nan, dtype=np.float64)
        spectral_change = np.full(number_of_bins, np.nan, dtype=np.float64)
        if number_of_bins > 1:
            valid_level_pair = np.isfinite(envelope[1:]) & np.isfinite(envelope[:-1])
            level_delta = np.abs(np.diff(envelope))
            level_change[1:][valid_level_pair] = level_delta[valid_level_pair]
            spectral_proxy = activity - envelope
            valid_spectral_pair = np.isfinite(spectral_proxy[1:]) & np.isfinite(
                spectral_proxy[:-1]
            )
            spectral_delta = np.abs(np.diff(spectral_proxy))
            spectral_change[1:][valid_spectral_pair] = spectral_delta[valid_spectral_pair]
        transition = level_change.copy()
        both_changes = np.isfinite(level_change) & np.isfinite(spectral_change)
        transition[both_changes] = (
            level_change[both_changes] + 0.35 * spectral_change[both_changes]
        )
        only_spectral = ~np.isfinite(level_change) & np.isfinite(spectral_change)
        transition[only_spectral] = 0.35 * spectral_change[only_spectral]
        transition = _nan_centered_smooth(
            transition, max(1, int(round(0.03 * rate_hz)))
        )

        finite_envelope = np.isfinite(envelope)
        noise_floor_dbfs: float | None = None
        speech_threshold_dbfs: float | None = None
        speech_mask = np.zeros(number_of_bins, dtype=bool)
        if np.count_nonzero(finite_envelope) >= max(10, int(rate_hz)):
            values = envelope[finite_envelope]
            noise_floor_dbfs = float(np.percentile(values, 15.0))
            high_level = float(np.percentile(values, 95.0))
            dynamic_range = max(0.0, high_level - noise_floor_dbfs)
            speech_threshold_dbfs = noise_floor_dbfs + float(
                np.clip(0.30 * dynamic_range, 6.0, 18.0)
            )
            speech_mask = finite_envelope & (envelope >= speech_threshold_dbfs)
            speech_mask = _close_short_boolean_gaps(
                speech_mask, max(1, int(round(0.08 * rate_hz)))
            )
            speech_mask = _remove_short_boolean_runs(
                speech_mask, max(1, int(round(0.06 * rate_hz)))
            )
            speech_mask &= finite_envelope

        valid_bins = int(np.count_nonzero(finite_envelope))
        diagnostics = AudioDiagnostics(
            stream_index=int(audio.index),
            source_sample_rate=(
                int(audio.codec_context.sample_rate)
                if getattr(audio.codec_context, "sample_rate", None)
                else None
            ),
            target_sample_rate=sample_rate,
            feature_rate_hz=rate_hz,
            start_s=start_s,
            end_s=effective_end_s,
            decoded_frames=decoded_frames,
            resampled_frames=resampled_frames,
            input_frames_without_pts=input_without_pts,
            output_frames_without_pts=output_without_pts,
            pts_reversals=pts_reversals,
            discontinuity_gaps=discontinuity_gaps,
            overlapping_chunks=overlapping_chunks,
            total_bins=number_of_bins,
            valid_bins=valid_bins,
            coverage_fraction=valid_bins / number_of_bins,
            noise_floor_dbfs=noise_floor_dbfs,
            speech_threshold_dbfs=speech_threshold_dbfs,
            warnings=tuple(warnings),
        )
        grid_s = start_s + (np.arange(number_of_bins, dtype=np.float64) + 0.5) / rate_hz
        return AudioFeatures(
            grid_s=grid_s,
            envelope=envelope,
            transition=transition,
            speech_mask=speech_mask,
            activity=activity,
            diagnostics=diagnostics,
        )


def save_rgb_thumbnail(
    rgb: np.ndarray,
    output_path: PathLike,
    *,
    max_width: int = 960,
    quality: int = 88,
) -> Path:
    """Save an RGB frame as a bounded-size thumbnail."""
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 array with shape (height, width, 3)")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array)
    if image.width > max_width:
        height = max(1, int(round(image.height * max_width / image.width)))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(destination, format="JPEG", quality=quality, optimize=True)
    elif suffix == ".webp":
        image.save(destination, format="WEBP", quality=quality, method=6)
    else:
        image.save(destination, format="PNG", optimize=True)
    return destination


__all__ = [
    "AudioDiagnostics",
    "AudioFeatures",
    "MediaError",
    "NoAudioStreamError",
    "NoVideoStreamError",
    "VideoFrameSample",
    "decode_audio_features",
    "iter_sampled_video_frames",
    "probe_media",
    "save_rgb_thumbnail",
]
