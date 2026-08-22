"""Deterministic descriptive acoustic features without biometric models.

This module intentionally provides no speaker embedding, voiceprint, speaker
verification, cross-clip similarity, identity score, or aggregate ranking.
Caller-supplied group and language labels are metadata, never signal outputs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 48_000
    frame_ms: float = 40.0
    hop_ms: float = 10.0
    f0_min_hz: float = 80.0
    f0_max_hz: float = 300.0
    minimum_periodicity: float = 0.35
    mel_bands: int = 64
    mel_min_hz: float = 50.0
    mel_max_hz: float = 12_000.0
    spectral_bands_hz: tuple[tuple[str, float, float], ...] = (
        ("low_0_500", 0.0, 500.0),
        ("mid_500_2000", 500.0, 2_000.0),
        ("high_2000_8000", 2_000.0, 8_000.0),
        ("upper_8000_nyquist", 8_000.0, 24_000.0),
    )

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0 or self.hop_ms <= 0 or self.hop_ms > self.frame_ms:
            raise ValueError("frame/hop duration is invalid")
        if not 0 < self.f0_min_hz < self.f0_max_hz < self.sample_rate / 2:
            raise ValueError("F0 range must lie inside 0..Nyquist")
        if not 0.0 <= self.minimum_periodicity <= 1.0:
            raise ValueError("minimum_periodicity must be within 0..1")
        if self.mel_bands < 4:
            raise ValueError("mel_bands must be at least 4")


@dataclass(frozen=True)
class DecodedAudio:
    path: Path
    samples: np.ndarray
    sample_rate: int
    source_sample_rate: int | None
    source_channels: int
    analysis_channels: int
    source_codec: str | None
    source_time_base: str | None
    source_stream_start_raw_s: float | None
    decoded_frames: int
    input_frames_without_pts: int
    requested_start_s: float
    requested_end_s: float | None


@dataclass(frozen=True)
class FeatureBundle:
    samples: np.ndarray
    sample_rate: int
    frame_times_s: np.ndarray
    frame_columns: dict[str, np.ndarray]
    mel_frequencies_hz: np.ndarray
    log_mel_power_db: np.ndarray
    summary: dict[str, Any]


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_audio(
    path: str | Path,
    *,
    sample_rate: int = 48_000,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> DecodedAudio:
    """Decode a local PTS-defined interval with PyAV, preserving mono/stereo."""

    import av

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(start_s) or start_s < 0.0:
        raise ValueError("start_s must be finite and non-negative")
    if end_s is not None and (not math.isfinite(end_s) or end_s <= start_s):
        raise ValueError("end_s must be finite and greater than start_s")

    with av.open(str(source), mode="r") as container:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            raise ValueError(f"no audio stream found in {source}")
        context = stream.codec_context
        source_rate = getattr(context, "sample_rate", None)
        source_channels = int(getattr(context, "channels", 0) or 0)
        if source_channels <= 0 and getattr(context, "layout", None) is not None:
            source_channels = len(context.layout.channels)
        if source_channels <= 0:
            source_channels = 1
        analysis_channels = 2 if source_channels >= 2 else 1
        layout = "stereo" if analysis_channels == 2 else "mono"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        blocks: list[np.ndarray] = []
        decoded_frames = 0
        frames_without_pts = 0
        if container.start_time is not None:
            timeline_origin_s = float(container.start_time / av.time_base)
        elif stream.start_time is not None and stream.time_base is not None:
            timeline_origin_s = float(stream.start_time * stream.time_base)
        else:
            timeline_origin_s = 0.0
        if start_s > 0.0:
            raw_seek_s = max(0.0, start_s - 0.5) + timeline_origin_s
            if stream.time_base is not None:
                container.seek(
                    int(math.floor(raw_seek_s / float(stream.time_base))),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
            else:
                container.seek(
                    int(math.floor(raw_seek_s * av.time_base)),
                    backward=True,
                    any_frame=False,
                )

        last_output_end_s: float | None = None

        def append_frame(frame: Any) -> None:
            nonlocal last_output_end_s
            array = np.asarray(frame.to_ndarray(), dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2:
                raise ValueError(f"unexpected decoded audio shape: {array.shape}")
            if array.shape[0] != analysis_channels:
                if array.size % analysis_channels != 0:
                    raise ValueError(f"cannot reshape decoded audio: {array.shape}")
                array = array.reshape(analysis_channels, -1)
            block = np.ascontiguousarray(array.T)
            if start_s == 0.0 and end_s is None:
                blocks.append(block)
                return
            if frame.pts is not None and frame.time_base is not None:
                block_start_s = float(frame.pts * frame.time_base) - timeline_origin_s
            elif last_output_end_s is not None:
                block_start_s = last_output_end_s
            else:
                raise ValueError(
                    "cannot trim audio segment because the first resampled frame has no PTS"
                )
            block_end_s = block_start_s + len(block) / sample_rate
            last_output_end_s = block_end_s
            selection_start_s = max(start_s, block_start_s)
            selection_end_s = min(end_s, block_end_s) if end_s is not None else block_end_s
            if selection_end_s <= selection_start_s:
                return
            first = max(
                0,
                int(math.ceil((selection_start_s - block_start_s) * sample_rate - 1e-7)),
            )
            last = min(
                len(block),
                int(math.ceil((selection_end_s - block_start_s) * sample_rate - 1e-7)),
            )
            if last > first:
                blocks.append(block[first:last])

        for frame in container.decode(stream):
            decoded_frames += 1
            if frame.pts is None:
                frames_without_pts += 1
            for output in resampler.resample(frame):
                append_frame(output)
            if (
                end_s is not None
                and frame.pts is not None
                and frame.time_base is not None
                and float(frame.pts * frame.time_base) - timeline_origin_s > end_s + 0.25
            ):
                break
        for output in resampler.resample(None):
            append_frame(output)

        if not blocks:
            raise ValueError(f"audio stream decoded no samples: {source}")
        samples = np.concatenate(blocks, axis=0).astype(np.float64, copy=False)
        if not np.all(np.isfinite(samples)):
            raise ValueError("decoded audio contains non-finite samples")
        stream_start_s = (
            float(stream.start_time * stream.time_base)
            if stream.start_time is not None and stream.time_base is not None
            else None
        )
        return DecodedAudio(
            path=source,
            samples=samples,
            sample_rate=sample_rate,
            source_sample_rate=int(source_rate) if source_rate else None,
            source_channels=source_channels,
            analysis_channels=analysis_channels,
            source_codec=str(getattr(context, "name", "") or "") or None,
            source_time_base=str(stream.time_base) if stream.time_base is not None else None,
            source_stream_start_raw_s=stream_start_s,
            decoded_frames=decoded_frames,
            input_frames_without_pts=frames_without_pts,
            requested_start_s=float(start_s),
            requested_end_s=float(end_s) if end_s is not None else None,
        )


def _frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("samples must have shape (sample, channel)")
    if samples.shape[0] < frame_length:
        samples = np.pad(samples, ((0, frame_length - samples.shape[0]), (0, 0)))
    frame_count = 1 + (samples.shape[0] - frame_length) // hop_length
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    indices = starts[:, None] + np.arange(frame_length, dtype=np.int64)[None, :]
    return samples[indices]


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    frequencies_hz: np.ndarray,
    *,
    bands: int,
    minimum_hz: float,
    maximum_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_hz = min(float(maximum_hz), float(frequencies_hz[-1]))
    if not 0.0 <= minimum_hz < maximum_hz:
        raise ValueError("invalid mel frequency range")
    points_hz = _mel_to_hz(
        np.linspace(_hz_to_mel(minimum_hz), _hz_to_mel(maximum_hz), bands + 2)
    )
    filters = np.zeros((bands, len(frequencies_hz)), dtype=np.float64)
    for index in range(bands):
        left, center, right = points_hz[index : index + 3]
        rising = (frequencies_hz - left) / max(center - left, EPSILON)
        falling = (right - frequencies_hz) / max(right - center, EPSILON)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
        total = float(np.sum(filters[index]))
        if total > EPSILON:
            filters[index] /= total
    return filters, points_hz[1:-1]


def normalized_autocorrelation_f0(
    mono_frames: np.ndarray,
    *,
    sample_rate: int,
    minimum_hz: float,
    maximum_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized-autocorrelation F0 candidates and peak periodicity."""

    if mono_frames.ndim != 2:
        raise ValueError("mono_frames must be two-dimensional")
    frame_count, frame_length = mono_frames.shape
    minimum_lag = max(1, int(math.floor(sample_rate / maximum_hz)))
    maximum_lag = min(frame_length - 2, int(math.ceil(sample_rate / minimum_hz)))
    if minimum_lag >= maximum_lag:
        raise ValueError("frame is too short for the configured F0 range")
    centered = mono_frames - np.mean(mono_frames, axis=1, keepdims=True)
    windowed = centered * np.hanning(frame_length)[None, :]
    fft_size = 1 << int(math.ceil(math.log2(2 * frame_length - 1)))
    spectrum = np.fft.rfft(windowed, n=fft_size, axis=1)
    autocorrelation = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=fft_size, axis=1
    )[:, :frame_length]
    squared = windowed * windowed
    cumulative = np.concatenate(
        (np.zeros((frame_count, 1)), np.cumsum(squared, axis=1)), axis=1
    )
    lags = np.arange(minimum_lag, maximum_lag + 1, dtype=np.int64)
    left_energy = cumulative[:, frame_length - lags]
    right_energy = cumulative[:, frame_length : frame_length + 1] - cumulative[:, lags]
    denominator = np.sqrt(np.maximum(left_energy * right_energy, EPSILON))
    correlation = np.clip(autocorrelation[:, lags] / denominator, -1.0, 1.0)
    best_indices = np.argmax(correlation, axis=1)
    best_lags = lags[best_indices].astype(np.float64)
    periodicity = correlation[np.arange(frame_count), best_indices]

    interior = (best_indices > 0) & (best_indices < len(lags) - 1)
    rows = np.nonzero(interior)[0]
    if rows.size:
        indices = best_indices[rows]
        left = correlation[rows, indices - 1]
        center = correlation[rows, indices]
        right = correlation[rows, indices + 1]
        curvature = left - 2.0 * center + right
        offsets = np.zeros_like(center)
        stable = np.abs(curvature) > 1e-9
        offsets[stable] = 0.5 * (left[stable] - right[stable]) / curvature[stable]
        best_lags[rows] += np.clip(offsets, -0.5, 0.5)
    return (
        sample_rate / np.maximum(best_lags, 1.0),
        np.clip(periodicity, 0.0, 1.0),
    )


def _finite_percentiles(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "iqr": None,
        }
    p05, p25, median, p75, p95 = np.percentile(finite, [5, 25, 50, 75, 95])
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p05": float(p05),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p95": float(p95),
        "iqr": float(p75 - p25),
    }


def analyze_signal(
    samples: np.ndarray,
    *,
    sample_rate: int,
    config: FeatureConfig | None = None,
) -> FeatureBundle:
    """Measure one clip independently; no cross-clip representation is built."""

    config = config or FeatureConfig(sample_rate=sample_rate)
    config.validate()
    if sample_rate != config.sample_rate:
        raise ValueError("samples must already use config.sample_rate")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise ValueError("samples must be mono or stereo")
    if samples.shape[0] < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("audio samples are missing or non-finite")

    frame_length = int(round(sample_rate * config.frame_ms / 1000.0))
    hop_length = int(round(sample_rate * config.hop_ms / 1000.0))
    frames = _frame_signal(samples, frame_length, hop_length)
    mono_frames = np.mean(frames, axis=2)
    frame_times_s = (
        np.arange(len(frames), dtype=np.float64) * hop_length + frame_length / 2
    ) / sample_rate
    rms = np.sqrt(np.mean(mono_frames * mono_frames, axis=1))
    rms_dbfs = 20.0 * np.log10(np.maximum(rms, EPSILON))
    activity_threshold_dbfs = min(
        -30.0, max(-55.0, float(np.percentile(rms_dbfs, 10.0)) + 10.0)
    )
    active = rms_dbfs >= activity_threshold_dbfs

    candidate_f0_hz, periodicity = normalized_autocorrelation_f0(
        mono_frames,
        sample_rate=sample_rate,
        minimum_hz=config.f0_min_hz,
        maximum_hz=config.f0_max_hz,
    )
    voiced = active & (periodicity >= config.minimum_periodicity)
    f0_hz = np.where(voiced, candidate_f0_hz, np.nan)

    window = np.hanning(frame_length)
    fft_size = 1 << int(math.ceil(math.log2(frame_length)))
    frequencies_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(mono_frames * window[None, :], n=fft_size, axis=1)
    power = np.abs(spectrum) ** 2 / max(float(np.sum(window * window)), EPSILON)
    total_power = np.maximum(np.sum(power, axis=1), EPSILON)
    centroid_hz = np.sum(power * frequencies_hz[None, :], axis=1) / total_power
    bandwidth_hz = np.sqrt(
        np.sum(
            power * (frequencies_hz[None, :] - centroid_hz[:, None]) ** 2,
            axis=1,
        )
        / total_power
    )
    cumulative_power = np.cumsum(power, axis=1)
    rolloff_indices = np.argmax(
        cumulative_power >= 0.85 * total_power[:, None], axis=1
    )
    rolloff_85_hz = frequencies_hz[rolloff_indices]
    flatness = np.exp(np.mean(np.log(power + EPSILON), axis=1)) / np.maximum(
        np.mean(power, axis=1), EPSILON
    )

    frame_columns: dict[str, np.ndarray] = {
        "time_s": frame_times_s,
        "rms_dbfs": rms_dbfs,
        "active": active,
        "f0_hz": f0_hz,
        "periodicity": periodicity,
        "voiced": voiced,
        "spectral_centroid_hz": centroid_hz,
        "spectral_bandwidth_hz": bandwidth_hz,
        "spectral_rolloff_85_hz": rolloff_85_hz,
        "spectral_flatness": flatness,
    }
    band_summary: dict[str, float | None] = {}
    nyquist = sample_rate / 2.0
    for name, lower_hz, upper_hz in config.spectral_bands_hz:
        effective_upper = min(upper_hz, nyquist)
        mask = (frequencies_hz >= lower_hz) & (
            frequencies_hz < effective_upper
            if effective_upper < nyquist
            else frequencies_hz <= effective_upper
        )
        fraction = (
            np.sum(power[:, mask], axis=1) / total_power
            if np.any(mask)
            else np.full(len(frames), np.nan)
        )
        frame_columns[f"band_fraction_{name}"] = fraction
        active_values = fraction[active & np.isfinite(fraction)]
        band_summary[name] = (
            float(np.mean(active_values)) if active_values.size else None
        )

    stereo_summary: dict[str, Any] = {"available": samples.shape[1] == 2}
    if samples.shape[1] == 2:
        left = frames[:, :, 0]
        right = frames[:, :, 1]
        left_rms = np.sqrt(np.mean(left * left, axis=1))
        right_rms = np.sqrt(np.mean(right * right, axis=1))
        left_rms_dbfs = 20.0 * np.log10(np.maximum(left_rms, EPSILON))
        right_rms_dbfs = 20.0 * np.log10(np.maximum(right_rms, EPSILON))
        balance_db = 20.0 * np.log10(
            np.maximum(left_rms, EPSILON) / np.maximum(right_rms, EPSILON)
        )
        left_centered = left - np.mean(left, axis=1, keepdims=True)
        right_centered = right - np.mean(right, axis=1, keepdims=True)
        correlation_denominator = np.sqrt(
            np.sum(left_centered * left_centered, axis=1)
            * np.sum(right_centered * right_centered, axis=1)
        )
        correlation = np.divide(
            np.sum(left_centered * right_centered, axis=1),
            correlation_denominator,
            out=np.full(len(frames), np.nan),
            where=correlation_denominator > EPSILON,
        )
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        mid_energy = np.mean(mid * mid, axis=1)
        side_energy = np.mean(side * side, axis=1)
        side_fraction = side_energy / np.maximum(mid_energy + side_energy, EPSILON)
        frame_columns.update(
            left_rms_dbfs=left_rms_dbfs,
            right_rms_dbfs=right_rms_dbfs,
            stereo_balance_db=balance_db,
            stereo_correlation=correlation,
            stereo_side_fraction=side_fraction,
        )
        stereo_summary.update(
            left_rms_dbfs=_finite_percentiles(left_rms_dbfs[active]),
            right_rms_dbfs=_finite_percentiles(right_rms_dbfs[active]),
            balance_db=_finite_percentiles(balance_db[active]),
            correlation=_finite_percentiles(correlation[active]),
            side_fraction=_finite_percentiles(side_fraction[active]),
        )
    else:
        nan_values = np.full(len(frames), np.nan)
        frame_columns.update(
            left_rms_dbfs=nan_values.copy(),
            right_rms_dbfs=nan_values.copy(),
            stereo_balance_db=nan_values.copy(),
            stereo_correlation=nan_values.copy(),
            stereo_side_fraction=nan_values.copy(),
        )

    mel_filters, mel_centers_hz = _mel_filterbank(
        frequencies_hz,
        bands=config.mel_bands,
        minimum_hz=config.mel_min_hz,
        maximum_hz=config.mel_max_hz,
    )
    log_mel_power_db = 10.0 * np.log10(power @ mel_filters.T + EPSILON)

    mono = np.mean(samples, axis=1)
    active_count = int(np.sum(active))
    voiced_count = int(np.sum(voiced))
    summary = {
        "duration_s": float(samples.shape[0] / sample_rate),
        "analysis_sample_rate": sample_rate,
        "analysis_channels": int(samples.shape[1]),
        "frame_ms": config.frame_ms,
        "hop_ms": config.hop_ms,
        "frame_count": int(len(frames)),
        "activity": {
            "threshold_dbfs": activity_threshold_dbfs,
            "active_frame_count": active_count,
            "active_frame_fraction": float(active_count / len(frames)),
        },
        "waveform": {
            "rms_dbfs": float(
                20.0 * np.log10(max(float(np.sqrt(np.mean(mono * mono))), EPSILON))
            ),
            "peak_dbfs": float(
                20.0 * np.log10(max(float(np.max(np.abs(samples))), EPSILON))
            ),
            "sample_fraction_abs_ge_0_999": float(np.mean(np.abs(samples) >= 0.999)),
            "dc_offset_by_channel": [float(value) for value in np.mean(samples, axis=0)],
        },
        "f0_hz": {
            "method": "normalized_autocorrelation_peak",
            "range_hz": [config.f0_min_hz, config.f0_max_hz],
            "minimum_periodicity": config.minimum_periodicity,
            "voiced_frame_count": voiced_count,
            "voiced_fraction_of_active": (
                float(voiced_count / active_count) if active_count else None
            ),
            "distribution": _finite_percentiles(f0_hz),
        },
        "periodicity": _finite_percentiles(periodicity[active]),
        "spectral": {
            "centroid_hz": _finite_percentiles(centroid_hz[active]),
            "bandwidth_hz": _finite_percentiles(bandwidth_hz[active]),
            "rolloff_85_hz": _finite_percentiles(rolloff_85_hz[active]),
            "flatness": _finite_percentiles(flatness[active]),
            "mean_active_band_fractions": band_summary,
        },
        "stereo": stereo_summary,
    }
    return FeatureBundle(
        samples=samples,
        sample_rate=sample_rate,
        frame_times_s=frame_times_s,
        frame_columns=frame_columns,
        mel_frequencies_hz=mel_centers_hz,
        log_mel_power_db=log_mel_power_db,
        summary=summary,
    )
