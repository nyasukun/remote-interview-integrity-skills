"""Event-level bilabial release timing from audio and visible lip motion.

The routines in this module deliberately separate *selection* from
*measurement*: an ASR timestamp only defines a short search window.  The audio
release is then re-estimated from the waveform, while the visual release is
estimated independently from FaceMesh geometry at decoded video PTS.

This is a measurement aid, not a deepfake classifier.  A non-zero result can
also be caused by capture, conferencing, encoding, or playback timing.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import av
import numpy as np

from .face_process import analyze_face_frames_isolated
from .media import NoAudioStreamError, iter_sampled_video_frames, probe_media


PathLike = str | os.PathLike[str]

_MOUTH_CORNERS = (61, 291)
_CENTRAL_LIP_PAIRS = ((13, 14), (82, 87), (312, 317))


@dataclass(frozen=True)
class LipGeometry:
    """Roll-normalized central lip distances divided by mouth width."""

    mouth_width: float
    pair_apertures: tuple[float, float, float]
    median_aperture: float
    maximum_aperture: float
    aperture_spread: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_lip_aperture(landmarks_xy: np.ndarray) -> LipGeometry:
    """Measure visible central lip separation from MediaPipe landmark XY.

    Distances are projected perpendicular to the 61–291 mouth-corner axis and
    divided by its length.  Consequently translation, uniform scale, and
    in-plane head roll do not alter the result.
    """

    points = np.asarray(landmarks_xy, dtype=np.float64)
    required_index = max(
        *_MOUTH_CORNERS,
        *(index for pair in _CENTRAL_LIP_PAIRS for index in pair),
    )
    if points.ndim != 2 or points.shape[1] != 2 or len(points) <= required_index:
        raise ValueError("landmarks_xy must have shape (N, 2) with FaceMesh indices")
    relevant = np.asarray(
        [*_MOUTH_CORNERS, *(index for pair in _CENTRAL_LIP_PAIRS for index in pair)]
    )
    if not np.all(np.isfinite(points[relevant])):
        raise ValueError("required mouth landmarks must be finite")

    mouth_axis = points[_MOUTH_CORNERS[1]] - points[_MOUTH_CORNERS[0]]
    mouth_width = float(np.linalg.norm(mouth_axis))
    if mouth_width <= 1e-9:
        raise ValueError("mouth corners must not coincide")
    axis_y = np.asarray((-mouth_axis[1], mouth_axis[0]), dtype=np.float64) / mouth_width
    apertures = np.asarray(
        [
            abs(float(np.dot(points[lower] - points[upper], axis_y))) / mouth_width
            for upper, lower in _CENTRAL_LIP_PAIRS
        ],
        dtype=np.float64,
    )
    return LipGeometry(
        mouth_width=mouth_width,
        pair_apertures=tuple(float(value) for value in apertures),  # type: ignore[arg-type]
        median_aperture=float(np.median(apertures)),
        maximum_aperture=float(np.max(apertures)),
        aperture_spread=float(np.ptp(apertures)),
    )


@dataclass(frozen=True)
class AudioWindow:
    """Short mono PCM window placed on the source container timeline."""

    start_s: float
    sample_rate: int
    waveform: np.ndarray
    coverage_mask: np.ndarray
    coverage_fraction: float
    warnings: tuple[str, ...] = ()

    @property
    def end_s(self) -> float:
        return self.start_s + len(self.waveform) / self.sample_rate


def _container_origin_s(container: av.container.InputContainer) -> float:
    if container.start_time is not None:
        return float(container.start_time / av.time_base)
    starts = [
        float(stream.start_time * stream.time_base)
        for stream in container.streams
        if stream.start_time is not None and stream.time_base is not None
    ]
    return min(starts, default=0.0)


def decode_audio_window(
    path: PathLike,
    *,
    start_s: float,
    end_s: float,
    sample_rate: int = 48_000,
) -> AudioWindow:
    """Decode a short PTS-aware mono window without inventing missing samples.

    Uncovered timeline samples are returned as NaN and disclosed through
    ``coverage_mask``.  Overlapping decoded chunks are averaged.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be finite and non-negative")
    if not math.isfinite(end_s) or end_s <= start_s:
        raise ValueError("end_s must be finite and greater than start_s")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    sample_count = int(math.ceil((end_s - start_s) * sample_rate - 1e-9))
    sums = np.zeros(sample_count, dtype=np.float64)
    counts = np.zeros(sample_count, dtype=np.uint16)
    warnings: list[str] = []

    with av.open(str(source), mode="r") as container:
        audio = next(iter(container.streams.audio), None)
        if audio is None:
            raise NoAudioStreamError(f"No audio stream found in {source}")
        origin_s = _container_origin_s(container)
        raw_seek_s = max(0.0, start_s + origin_s - 0.12)
        try:
            if audio.time_base is not None:
                container.seek(
                    int(math.floor(raw_seek_s / float(audio.time_base))),
                    stream=audio,
                    backward=True,
                    any_frame=False,
                )
            else:
                container.seek(int(raw_seek_s * av.time_base), backward=True)
        except Exception as error:  # format-specific PyAV exception subclasses
            warnings.append(f"audio seek failed: {type(error).__name__}: {error}")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        last_output_end_s: float | None = None
        done = False

        def accumulate(output: av.AudioFrame) -> None:
            nonlocal last_output_end_s
            values = np.asarray(output.to_ndarray(), dtype=np.float64).reshape(-1)
            if not values.size:
                return
            if output.pts is not None and output.time_base is not None:
                chunk_start_s = float(output.pts * output.time_base) - origin_s
            elif last_output_end_s is not None:
                chunk_start_s = last_output_end_s
                warnings.append("resampled audio frame lacked PTS; contiguous time was used")
            else:
                warnings.append("resampled audio frame lacked PTS and was skipped")
                return
            last_output_end_s = chunk_start_s + len(values) / sample_rate

            first = int(round((chunk_start_s - start_s) * sample_rate))
            source_first = max(0, -first)
            target_first = max(0, first)
            length = min(len(values) - source_first, sample_count - target_first)
            if length <= 0:
                return
            segment = values[source_first : source_first + length]
            finite = np.isfinite(segment)
            target = slice(target_first, target_first + length)
            sums[target] += np.where(finite, segment, 0.0)
            counts[target] += finite.astype(np.uint16)

        for packet in container.demux(audio):
            for frame in packet.decode():
                if frame.pts is not None and frame.time_base is not None:
                    frame_start_s = float(frame.pts * frame.time_base) - origin_s
                    if frame_start_s > end_s + 0.15:
                        done = True
                        break
                for output in resampler.resample(frame):
                    accumulate(output)
            if done:
                break
        for output in resampler.resample(None):
            accumulate(output)

    coverage = counts > 0
    waveform = np.full(sample_count, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=waveform, where=coverage)
    return AudioWindow(
        start_s=float(start_s),
        sample_rate=int(sample_rate),
        waveform=waveform,
        coverage_mask=coverage,
        coverage_fraction=float(np.mean(coverage)) if coverage.size else 0.0,
        warnings=tuple(dict.fromkeys(warnings)),
    )


@dataclass(frozen=True)
class AcousticReleaseConfig:
    # The wide search retains alternatives for audio-blinded manual review.
    # Conservative auto-acceptance below uses a narrower anchor-offset gate.
    frame_ms: float = 8.0
    hop_ms: float = 1.0
    fft_size: int = 1_024
    search_before_s: float = 0.300
    search_after_s: float = 0.220
    normalization_before_s: float = 0.400
    normalization_after_s: float = 0.320
    context_before_s: float = 0.035
    context_after_s: float = 0.016
    minimum_separation_ms: float = 14.0
    minimum_coverage: float = 0.90
    candidate_minimum_energy_rise_db: float = 3.0
    candidate_minimum_energy_slope_12ms_db: float = 2.0
    candidate_minimum_post_energy_percentile: float = 18.0
    acceptance_mode: str = "permissive"
    # Legacy/permissive diagnostic acceptance gates.
    minimum_flux_z: float = 1.50
    minimum_high_frequency_rise_db: float = 1.50
    minimum_energy_rise_db_p: float = 2.50
    minimum_energy_rise_db_b: float = 0.50
    # Conservative candidate gates retained as uncalibrated starting values.
    # They are not a general validation claim. ASR probability is
    # applied upstream before this waveform-only measurement.
    conservative_minimum_energy_rise_db: float = 10.0
    conservative_minimum_energy_slope_12ms_db: float = 10.0
    conservative_minimum_high_frequency_rise_db: float = 8.0
    conservative_minimum_score: float = 5.5
    conservative_minimum_runner_up_margin: float = 1.5
    conservative_minimum_offset_ms: float = -60.0
    conservative_maximum_offset_ms: float = 160.0
    broadband_low_hz: float = 80.0
    spectral_high_hz: float = 12_000.0
    high_frequency_low_hz: float = 2_500.0
    flux_low_hz: float = 500.0
    maximum_candidates: int = 8

    @classmethod
    def conservative(cls, **overrides: Any) -> "AcousticReleaseConfig":
        """Create the strict candidate configuration for case-level calibration."""

        return cls(acceptance_mode="conservative", **overrides)


@dataclass(frozen=True)
class AcousticReleaseCandidate:
    time_s: float
    score: float
    spectral_flux_z: float
    high_frequency_rise_db: float
    energy_rise_db: float
    energy_slope_12ms_db: float
    peak_rms_dbfs: float
    pre_rms_dbfs: float
    post_rms_dbfs: float
    local_coverage: float
    distance_from_anchor_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcousticReleaseEstimate:
    anchor_time_s: float
    phone_class: str
    candidate_time_s: float | None
    release_time_s: float | None
    score: float | None
    runner_up_margin: float | None
    acceptance_mode: str
    confidence: str
    measurable: bool
    time_resolution_ms: float
    exclusion_reasons: tuple[str, ...]
    selected_candidate: AcousticReleaseCandidate | None
    candidates: tuple[AcousticReleaseCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.zeros(values.shape, dtype=np.float64)
    if np.count_nonzero(finite) < 3:
        return result
    center = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - center)))
    scale = max(1.4826 * mad, 1e-5)
    result[finite] = (values[finite] - center) / scale
    return result


def _median_in_range(
    values: np.ndarray,
    times_s: np.ndarray,
    lower_s: float,
    upper_s: float,
) -> float:
    selected = values[(times_s >= lower_s) & (times_s <= upper_s) & np.isfinite(values)]
    return float(np.median(selected)) if selected.size else math.nan


def _gaussian_smooth_edge(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian-smooth one feature while replicating its edge samples."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or sigma_frames <= 0:
        return values.copy()
    radius = max(1, int(math.ceil(4.0 * sigma_frames)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _offset_value(
    values: np.ndarray,
    times_s: np.ndarray,
    center_s: float,
    offset_s: float,
) -> float:
    return float(
        np.interp(
            center_s + offset_s,
            times_s,
            values,
            left=float(values[0]),
            right=float(values[-1]),
        )
    )


def rank_acoustic_release_candidates(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    window_start_s: float,
    anchor_time_s: float,
    coverage_mask: np.ndarray | None = None,
    config: AcousticReleaseConfig | None = None,
) -> tuple[AcousticReleaseCandidate, ...]:
    """Rank waveform burst/release candidates near a preselected ASR anchor."""

    config = config or AcousticReleaseConfig()
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or not math.isfinite(window_start_s) or not math.isfinite(anchor_time_s):
        raise ValueError("sample_rate and timestamps must be valid")
    if coverage_mask is None:
        covered = np.isfinite(values)
    else:
        covered = np.asarray(coverage_mask, dtype=bool).reshape(-1) & np.isfinite(values)
        if covered.shape != values.shape:
            raise ValueError("coverage_mask must match waveform")
    values = np.where(covered, values, 0.0)

    frame_length = max(16, int(round(config.frame_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(config.hop_ms * sample_rate / 1000.0)))
    if values.size < frame_length + hop:
        return ()
    frames = np.lib.stride_tricks.sliding_window_view(values, frame_length)[::hop]
    coverage_frames = np.lib.stride_tricks.sliding_window_view(
        covered.astype(np.float64), frame_length
    )[::hop]
    frame_coverage = np.mean(coverage_frames, axis=1)
    centers_s = (
        window_start_s
        + (np.arange(len(frames), dtype=np.float64) * hop + (frame_length - 1) / 2.0)
        / sample_rate
    )

    window = np.hanning(frame_length)
    raw_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-16)
    rms_db = 20.0 * np.log10(np.maximum(raw_rms, 1e-8))
    fft_size = max(int(config.fft_size), frame_length)
    spectrum = np.abs(np.fft.rfft(frames * window, n=fft_size, axis=1))
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    high_limit = min(config.spectral_high_hz, sample_rate * 0.48)
    broadband = (frequencies >= config.broadband_low_hz) & (frequencies <= high_limit)
    high_band = (frequencies >= config.high_frequency_low_hz) & (frequencies <= high_limit)
    flux_band = (frequencies >= config.flux_low_hz) & (frequencies <= high_limit)
    if min(np.count_nonzero(broadband), np.count_nonzero(high_band), np.count_nonzero(flux_band)) < 2:
        return ()

    broadband_db = 10.0 * np.log10(
        np.maximum(np.sum(spectrum[:, broadband] ** 2, axis=1), 1e-16)
    )
    high_db = 10.0 * np.log10(
        np.maximum(np.sum(spectrum[:, high_band] ** 2, axis=1), 1e-16)
    )
    hop_actual_ms = hop / sample_rate * 1000.0
    broadband_db = _gaussian_smooth_edge(
        broadband_db, sigma_frames=1.3 / hop_actual_ms
    )
    high_db = _gaussian_smooth_edge(high_db, sigma_frames=1.0 / hop_actual_ms)

    positive_delta = np.maximum(np.diff(spectrum[:, flux_band], axis=0), 0.0)
    flux_log_power = np.full(len(frames), -160.0, dtype=np.float64)
    flux_log_power[1:] = 10.0 * np.log10(
        np.maximum(np.sum(positive_delta**2, axis=1), 1e-16)
    )
    flux_log_power = _gaussian_smooth_edge(
        flux_log_power, sigma_frames=1.0 / hop_actual_ms
    )

    energy_rise = np.full(len(frames), np.nan, dtype=np.float64)
    energy_slope_12ms = np.full(len(frames), np.nan, dtype=np.float64)
    high_rise = np.full(len(frames), np.nan, dtype=np.float64)
    post_energy_level = np.full(len(frames), np.nan, dtype=np.float64)
    for index, time_s in enumerate(centers_s):
        pre_energy = _median_in_range(
            broadband_db, centers_s, time_s - 0.035, time_s - 0.008
        )
        post_energy = _median_in_range(
            broadband_db, centers_s, time_s + 0.004, time_s + 0.016
        )
        pre_high = _median_in_range(
            high_db, centers_s, time_s - 0.030, time_s - 0.007
        )
        post_high = _median_in_range(
            high_db, centers_s, time_s, time_s + 0.006
        )
        if np.isfinite(pre_energy) and np.isfinite(post_energy):
            energy_rise[index] = post_energy - pre_energy
            post_energy_level[index] = post_energy
        if np.isfinite(pre_high) and np.isfinite(post_high):
            high_rise[index] = post_high - pre_high
        energy_slope_12ms[index] = _offset_value(
            broadband_db, centers_s, time_s, 0.006
        ) - _offset_value(broadband_db, centers_s, time_s, -0.006)

    normalization = (
        (centers_s >= anchor_time_s - config.normalization_before_s)
        & (centers_s <= anchor_time_s + config.normalization_after_s)
    )
    normalization_indices = np.flatnonzero(normalization)
    if not normalization_indices.size:
        return ()
    search = (
        (centers_s >= anchor_time_s - config.search_before_s)
        & (centers_s <= anchor_time_s + config.search_after_s)
    )
    indices = np.flatnonzero(search)
    if not indices.size:
        return ()

    slope_z = np.clip(_robust_z(energy_slope_12ms[normalization]), -3.0, 5.0)
    rise_z = np.clip(_robust_z(energy_rise[normalization]), -3.0, 5.0)
    high_z = np.clip(_robust_z(high_rise[normalization]), -3.0, 5.0)
    flux_z = np.clip(_robust_z(flux_log_power[normalization]), -3.0, 5.0)
    unpenalized_scores = 0.95 * slope_z + 0.70 * rise_z + 0.45 * high_z + 0.35 * flux_z
    raw_score_by_index = np.full(len(frames), np.nan, dtype=np.float64)
    raw_score_by_index[normalization_indices] = unpenalized_scores
    offsets_ms = (centers_s[search] - anchor_time_s) * 1000.0
    score_by_index = np.full(len(frames), np.nan, dtype=np.float64)
    score_by_index[indices] = raw_score_by_index[indices] - 0.80 * np.abs(
        (offsets_ms - 45.0) / 150.0
    )
    flux_z_by_index = np.full(len(frames), np.nan, dtype=np.float64)
    flux_z_by_index[normalization_indices] = flux_z
    post_energy_floor = float(
        np.percentile(
            broadband_db[normalization],
            config.candidate_minimum_post_energy_percentile,
        )
    )

    raw_candidates: list[tuple[float, AcousticReleaseCandidate]] = []
    for index in indices:
        if index <= 0 or index >= len(raw_score_by_index) - 1:
            continue
        if (
            not np.isfinite(raw_score_by_index[index])
            or raw_score_by_index[index] < raw_score_by_index[index - 1]
            or raw_score_by_index[index] < raw_score_by_index[index + 1]
        ):
            continue
        if (
            energy_rise[index] < config.candidate_minimum_energy_rise_db
            or energy_slope_12ms[index]
            < config.candidate_minimum_energy_slope_12ms_db
            or post_energy_level[index] < post_energy_floor
        ):
            continue
        time_s = float(centers_s[index])
        pre_rms = _median_in_range(
            rms_db,
            centers_s,
            time_s - 0.035,
            time_s - 0.008,
        )
        post_rms = _median_in_range(
            rms_db,
            centers_s,
            time_s + 0.004,
            time_s + 0.016,
        )
        peak_rms_values = rms_db[
            (centers_s >= time_s - 0.004) & (centers_s <= time_s + 0.008)
        ]
        if not (
            np.isfinite(pre_rms)
            and np.isfinite(post_rms)
            and peak_rms_values.size
            and np.isfinite(energy_rise[index])
            and np.isfinite(energy_slope_12ms[index])
            and np.isfinite(high_rise[index])
        ):
            continue
        local_coverage = float(
            np.mean(
                frame_coverage[
                    (centers_s >= time_s - 0.035)
                    & (centers_s <= time_s + 0.016)
                ]
            )
        )
        distance_ms = (time_s - anchor_time_s) * 1000.0
        raw_candidates.append(
            (
                float(raw_score_by_index[index]),
                AcousticReleaseCandidate(
                time_s=time_s,
                score=float(score_by_index[index]),
                spectral_flux_z=float(flux_z_by_index[index]),
                high_frequency_rise_db=float(high_rise[index]),
                energy_rise_db=float(energy_rise[index]),
                energy_slope_12ms_db=float(energy_slope_12ms[index]),
                peak_rms_dbfs=float(np.max(peak_rms_values)),
                pre_rms_dbfs=pre_rms,
                post_rms_dbfs=post_rms,
                local_coverage=local_coverage,
                distance_from_anchor_ms=distance_ms,
                ),
            )
        )

    # Select peaks by the unpenalized physical score, then rank the survivors
    # by adjusted score. This prevents the ASR proximity penalty from moving a
    # peak boundary while still using it to identify the likely target mora.
    selected: list[tuple[float, AcousticReleaseCandidate]] = []
    minimum_separation_s = config.minimum_separation_ms / 1000.0
    for raw_score, candidate in sorted(raw_candidates, key=lambda item: item[0], reverse=True):
        if all(
            abs(candidate.time_s - other.time_s) >= minimum_separation_s
            for _, other in selected
        ):
            selected.append((raw_score, candidate))
    return tuple(
        candidate
        for _, candidate in sorted(
            selected, key=lambda item: item[1].score, reverse=True
        )[: config.maximum_candidates]
    )


def estimate_acoustic_release(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    window_start_s: float,
    anchor_time_s: float,
    phone_class: str = "p",
    coverage_mask: np.ndarray | None = None,
    config: AcousticReleaseConfig | None = None,
) -> AcousticReleaseEstimate:
    """Select a measurable /p/ or /b/ release from ranked burst candidates."""

    config = config or AcousticReleaseConfig()
    if config.acceptance_mode not in {"permissive", "conservative"}:
        raise ValueError("acceptance_mode must be 'permissive' or 'conservative'")
    normalized_phone = phone_class.lower().strip("/ ")
    if normalized_phone not in {"p", "b"}:
        raise ValueError("phone_class must be 'p' or 'b'")
    candidates = rank_acoustic_release_candidates(
        waveform,
        sample_rate,
        window_start_s=window_start_s,
        anchor_time_s=anchor_time_s,
        coverage_mask=coverage_mask,
        config=config,
    )
    if not candidates:
        return AcousticReleaseEstimate(
            anchor_time_s=float(anchor_time_s),
            phone_class=normalized_phone,
            candidate_time_s=None,
            release_time_s=None,
            score=None,
            runner_up_margin=None,
            acceptance_mode=config.acceptance_mode,
            confidence="insufficient",
            measurable=False,
            time_resolution_ms=config.hop_ms,
            exclusion_reasons=("no_acoustic_release_candidate",),
            selected_candidate=None,
            candidates=(),
        )

    runner_up_margin = (
        float(candidates[0].score - candidates[1].score)
        if len(candidates) >= 2
        else None
    )
    if config.acceptance_mode == "conservative":
        # Auto-accept only the unambiguous top-1 candidate. The complete ranked
        # list remains available for audio-blinded manual review when it fails.
        selected = candidates[0]
        reasons: list[str] = []
        if selected.local_coverage < config.minimum_coverage:
            reasons.append("incomplete_audio_coverage")
        if selected.energy_rise_db < config.conservative_minimum_energy_rise_db:
            reasons.append("conservative_energy_rise_below_threshold")
        if (
            selected.energy_slope_12ms_db
            < config.conservative_minimum_energy_slope_12ms_db
        ):
            reasons.append("conservative_12ms_energy_slope_below_threshold")
        if (
            selected.high_frequency_rise_db
            < config.conservative_minimum_high_frequency_rise_db
        ):
            reasons.append("conservative_high_frequency_rise_below_threshold")
        if selected.score < config.conservative_minimum_score:
            reasons.append("conservative_score_below_threshold")
        if (
            runner_up_margin is not None
            and runner_up_margin < config.conservative_minimum_runner_up_margin
        ):
            reasons.append("ambiguous_acoustic_top_candidate")
        if not (
            config.conservative_minimum_offset_ms
            <= selected.distance_from_anchor_ms
            <= config.conservative_maximum_offset_ms
        ):
            reasons.append("acoustic_candidate_outside_conservative_anchor_gate")
    else:
        required_energy_rise = (
            config.minimum_energy_rise_db_p
            if normalized_phone == "p"
            else config.minimum_energy_rise_db_b
        )

        def permissive_exclusions(
            candidate: AcousticReleaseCandidate,
        ) -> list[str]:
            candidate_reasons: list[str] = []
            if candidate.local_coverage < config.minimum_coverage:
                candidate_reasons.append("incomplete_audio_coverage")
            if candidate.spectral_flux_z < config.minimum_flux_z:
                candidate_reasons.append("weak_spectral_burst")
            if (
                candidate.high_frequency_rise_db
                < config.minimum_high_frequency_rise_db
            ):
                candidate_reasons.append("weak_high_frequency_release")
            if candidate.energy_rise_db < required_energy_rise:
                candidate_reasons.append("no_clear_post_release_energy_rise")
            return candidate_reasons

        selected = next(
            (
                candidate
                for candidate in candidates
                if not permissive_exclusions(candidate)
            ),
            candidates[0],
        )
        reasons = permissive_exclusions(selected)

    measurable = not reasons
    if not measurable:
        confidence = "insufficient"
    elif config.acceptance_mode == "conservative":
        confidence = "high"
    elif (
        selected.spectral_flux_z >= 4.0
        and selected.high_frequency_rise_db >= 6.0
        and selected.energy_rise_db >= 4.0
    ):
        confidence = "high"
    else:
        confidence = "medium"
    return AcousticReleaseEstimate(
        anchor_time_s=float(anchor_time_s),
        phone_class=normalized_phone,
        candidate_time_s=selected.time_s,
        release_time_s=selected.time_s if measurable else None,
        score=selected.score,
        runner_up_margin=runner_up_margin,
        acceptance_mode=config.acceptance_mode,
        confidence=confidence,
        measurable=measurable,
        time_resolution_ms=config.hop_ms,
        exclusion_reasons=tuple(reasons),
        selected_candidate=selected,
        candidates=candidates,
    )


@dataclass(frozen=True)
class LipApertureSample:
    time_s: float
    median_aperture: float
    maximum_aperture: float
    aperture_spread: float
    mouth_width_px: float
    face_quality: float
    pair_apertures: tuple[float, ...] = ()
    repeated_frame: bool = False
    repeat_distance: float | None = None
    face_detected: bool = True
    track_id: int | None = None
    source_pts: int | None = None
    source_time_base: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualReleaseConfig:
    median_filter_frames: int = 3
    closed_median_max: float = 0.015
    closed_pair_max: float = 0.025
    reopened_median_min: float = 0.035
    maximum_pair_spread: float = 0.020
    minimum_mouth_width_px: float = 80.0
    minimum_face_quality: float = 0.25
    maximum_mouth_width_jump_fraction: float = 0.15
    minimum_confirmation_frames: int = 2
    confirmation_window_s: float = 0.180
    minimum_closure_s: float = 0.080
    maximum_closure_s: float = 0.600
    search_before_s: float = 0.300
    search_after_s: float = 0.350
    maximum_release_gap_s: float = 0.110
    freeze_exclusion_s: float = 0.125


@dataclass(frozen=True)
class VisualReleaseEstimate:
    anchor_time_s: float
    contact_time_s: float | None
    candidate_time_s: float | None
    release_time_s: float | None
    closure_duration_ms: float | None
    timing_uncertainty_ms: float | None
    closed_median_aperture: float | None
    reopened_median_aperture: float | None
    valid_frame_fraction: float
    quality_score: float | None
    measurable: bool
    exclusion_reasons: tuple[str, ...]
    closed_frame_times_s: tuple[float, ...] = ()
    reopened_frame_times_s: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthShapeAtAcousticRelease:
    """Visible mouth geometry in the nearest real frame to the audio burst."""

    acoustic_release_time_s: float | None
    frame_time_s: float | None
    frame_time_error_ms: float | None
    category: str
    pair_apertures: tuple[float, ...]
    median_aperture: float | None
    maximum_aperture: float | None
    aperture_spread: float | None
    mouth_width_px: float | None
    face_quality: float | None
    repeated_frame: bool | None
    valid: bool
    exclusion_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def identify_mouth_shape_at_acoustic_release(
    samples: Sequence[LipApertureSample],
    acoustic: AcousticReleaseEstimate,
    *,
    config: VisualReleaseConfig | None = None,
) -> MouthShapeAtAcousticRelease:
    """Classify the mouth in the frame nearest the independently found burst."""

    config = config or VisualReleaseConfig()
    if not acoustic.measurable or acoustic.release_time_s is None:
        return MouthShapeAtAcousticRelease(
            acoustic_release_time_s=acoustic.release_time_s,
            frame_time_s=None,
            frame_time_error_ms=None,
            category="unavailable",
            pair_apertures=(),
            median_aperture=None,
            maximum_aperture=None,
            aperture_spread=None,
            mouth_width_px=None,
            face_quality=None,
            repeated_frame=None,
            valid=False,
            exclusion_reasons=("acoustic_release_not_measurable",),
        )
    if not samples:
        return MouthShapeAtAcousticRelease(
            acoustic_release_time_s=acoustic.release_time_s,
            frame_time_s=None,
            frame_time_error_ms=None,
            category="unavailable",
            pair_apertures=(),
            median_aperture=None,
            maximum_aperture=None,
            aperture_spread=None,
            mouth_width_px=None,
            face_quality=None,
            repeated_frame=None,
            valid=False,
            exclusion_reasons=("no_video_frame_near_acoustic_release",),
        )

    ordered = sorted(samples, key=lambda item: item.time_s)
    selected = min(
        ordered, key=lambda item: abs(item.time_s - float(acoustic.release_time_s))
    )
    error_ms = (selected.time_s - float(acoustic.release_time_s)) * 1000.0
    reasons: list[str] = []
    if not selected.face_detected:
        reasons.append("face_not_detected_at_acoustic_release")
    if not all(
        math.isfinite(value)
        for value in (
            selected.median_aperture,
            selected.maximum_aperture,
            selected.aperture_spread,
            selected.mouth_width_px,
            selected.face_quality,
        )
    ):
        reasons.append("nonfinite_mouth_geometry_at_acoustic_release")
    if (
        math.isfinite(selected.mouth_width_px)
        and selected.mouth_width_px < config.minimum_mouth_width_px
    ):
        reasons.append("mouth_resolution_too_low_at_acoustic_release")
    if (
        math.isfinite(selected.aperture_spread)
        and selected.aperture_spread > config.maximum_pair_spread
    ):
        reasons.append("inconsistent_lip_landmarks_at_acoustic_release")
    if (
        math.isfinite(selected.face_quality)
        and selected.face_quality < config.minimum_face_quality
    ):
        reasons.append("low_face_quality_at_acoustic_release")
    if len(ordered) >= 2:
        frame_period_s = float(np.median(np.diff([item.time_s for item in ordered])))
        if abs(error_ms) > max(50.0, frame_period_s * 750.0):
            reasons.append("no_video_frame_near_acoustic_release")

    if reasons:
        category = "unavailable"
    elif (
        selected.median_aperture <= config.closed_median_max
        and selected.maximum_aperture <= config.closed_pair_max
    ):
        category = "closed/contact"
    elif selected.median_aperture >= config.reopened_median_min:
        category = "open"
    else:
        category = "transition"
    return MouthShapeAtAcousticRelease(
        acoustic_release_time_s=acoustic.release_time_s,
        frame_time_s=selected.time_s,
        frame_time_error_ms=error_ms,
        category=category,
        pair_apertures=tuple(float(value) for value in selected.pair_apertures),
        median_aperture=(
            float(selected.median_aperture)
            if math.isfinite(selected.median_aperture)
            else None
        ),
        maximum_aperture=(
            float(selected.maximum_aperture)
            if math.isfinite(selected.maximum_aperture)
            else None
        ),
        aperture_spread=(
            float(selected.aperture_spread)
            if math.isfinite(selected.aperture_spread)
            else None
        ),
        mouth_width_px=(
            float(selected.mouth_width_px)
            if math.isfinite(selected.mouth_width_px)
            else None
        ),
        face_quality=(
            float(selected.face_quality)
            if math.isfinite(selected.face_quality)
            else None
        ),
        repeated_frame=bool(selected.repeated_frame),
        valid=not reasons,
        exclusion_reasons=tuple(dict.fromkeys(reasons)),
    )


def _centered_nanmedian(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    if width % 2 != 1:
        raise ValueError("median_filter_frames must be odd")
    half = width // 2
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(len(values)):
        local = values[max(0, index - half) : min(len(values), index + half + 1)]
        finite = local[np.isfinite(local)]
        if finite.size >= max(1, width // 2 + 1):
            result[index] = float(np.median(finite))
    return result


def _long_freeze_before_release(
    samples: Sequence[LipApertureSample],
    release_index: int,
) -> float:
    if release_index <= 0:
        return 0.0
    last = release_index - 1
    if not samples[last].repeated_frame:
        return 0.0
    first = last
    while first > 0 and samples[first].repeated_frame:
        first -= 1
    return max(0.0, samples[last].time_s - samples[first].time_s)


def estimate_visual_release(
    samples: Sequence[LipApertureSample],
    *,
    anchor_time_s: float,
    config: VisualReleaseConfig | None = None,
) -> VisualReleaseEstimate:
    """Find a quality-controlled visible open→contact→reopen sequence."""

    config = config or VisualReleaseConfig()
    if not math.isfinite(anchor_time_s):
        raise ValueError("anchor_time_s must be finite")
    ordered = sorted(samples, key=lambda item: item.time_s)
    if len(ordered) < 5:
        return VisualReleaseEstimate(
            anchor_time_s=float(anchor_time_s),
            contact_time_s=None,
            candidate_time_s=None,
            release_time_s=None,
            closure_duration_ms=None,
            timing_uncertainty_ms=None,
            closed_median_aperture=None,
            reopened_median_aperture=None,
            valid_frame_fraction=0.0,
            quality_score=None,
            measurable=False,
            exclusion_reasons=("insufficient_video_frames",),
        )
    times = np.asarray([item.time_s for item in ordered], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("sample timestamps must be finite and unique")
    median_aperture = np.asarray([item.median_aperture for item in ordered])
    maximum_aperture = np.asarray([item.maximum_aperture for item in ordered])
    spread = np.asarray([item.aperture_spread for item in ordered])
    mouth_width = np.asarray([item.mouth_width_px for item in ordered])
    quality = np.asarray([item.face_quality for item in ordered])
    detected = np.asarray([item.face_detected for item in ordered], dtype=bool)

    width_jump = np.zeros(len(ordered), dtype=bool)
    finite_width = np.isfinite(mouth_width) & (mouth_width > 0)
    for index in range(1, len(ordered)):
        if finite_width[index] and finite_width[index - 1]:
            denominator = max(mouth_width[index], mouth_width[index - 1], 1e-9)
            if abs(mouth_width[index] - mouth_width[index - 1]) / denominator > config.maximum_mouth_width_jump_fraction:
                width_jump[index] = True
                width_jump[index - 1] = True
    valid = (
        detected
        & np.isfinite(median_aperture)
        & np.isfinite(maximum_aperture)
        & np.isfinite(spread)
        & finite_width
        & np.isfinite(quality)
        & (quality >= config.minimum_face_quality)
        & (mouth_width >= config.minimum_mouth_width_px)
        & (spread <= config.maximum_pair_spread)
        & ~width_jump
    )
    filtered_median = _centered_nanmedian(
        np.where(valid, median_aperture, np.nan), config.median_filter_frames
    )
    filtered_maximum = _centered_nanmedian(
        np.where(valid, maximum_aperture, np.nan), config.median_filter_frames
    )
    classified = valid & np.isfinite(filtered_median) & np.isfinite(filtered_maximum)
    closed = (
        classified
        & (filtered_median <= config.closed_median_max)
        & (filtered_maximum <= config.closed_pair_max)
    )
    reopened = classified & (filtered_median >= config.reopened_median_min)
    nonrepeat = ~np.asarray([item.repeated_frame for item in ordered], dtype=bool)
    valid_fraction = float(np.mean(valid))
    frame_period_s = float(np.median(np.diff(times)))

    possible: list[VisualReleaseEstimate] = []
    index = 0
    while index < len(ordered):
        if not closed[index]:
            index += 1
            continue
        run_start = index
        while index + 1 < len(ordered) and closed[index + 1]:
            index += 1
        run_end = index
        release_index = run_end + 1
        while release_index < len(ordered) and not classified[release_index]:
            release_index += 1
        if release_index >= len(ordered):
            break
        release_time = float(times[release_index])
        if not (
            anchor_time_s - config.search_before_s
            <= release_time
            <= anchor_time_s + config.search_after_s
        ):
            index += 1
            continue

        reasons: list[str] = []
        closed_indices = np.arange(run_start, run_end + 1)
        closed_confirmed = closed_indices[nonrepeat[closed_indices]]
        if len(closed_confirmed) < config.minimum_confirmation_frames:
            reasons.append("insufficient_nonrepeated_closed_frames")

        post_indices = np.flatnonzero(
            reopened
            & nonrepeat
            & (times >= release_time)
            & (times <= release_time + config.confirmation_window_s)
        )
        if len(post_indices) < config.minimum_confirmation_frames:
            reasons.append("reopening_not_confirmed")
        pre_indices = np.flatnonzero(
            reopened
            & nonrepeat
            & (times < times[run_start])
            & (times >= times[run_start] - config.maximum_closure_s)
        )
        if len(pre_indices) < config.minimum_confirmation_frames:
            reasons.append("preclosure_opening_not_confirmed")

        prior_index = run_start - 1
        while prior_index >= 0 and not classified[prior_index]:
            prior_index -= 1
        if prior_index >= 0:
            contact_time = float((times[prior_index] + times[run_start]) * 0.5)
        else:
            contact_time = float(times[run_start] - frame_period_s * 0.5)
        closure_s = release_time - contact_time
        if closure_s < config.minimum_closure_s:
            reasons.append("closure_too_short")
        if closure_s > config.maximum_closure_s:
            reasons.append("closure_too_long_or_pause")

        edge_gap_s = release_time - float(times[run_end])
        if edge_gap_s > config.maximum_release_gap_s:
            reasons.append("video_gap_at_release")
        freeze_s = _long_freeze_before_release(ordered, release_index)
        if freeze_s >= config.freeze_exclusion_s:
            reasons.append("long_repeated_frame_run_crosses_release")

        confirmed_post = post_indices[: config.minimum_confirmation_frames]
        closed_values = filtered_median[closed_indices]
        reopened_values = filtered_median[confirmed_post]
        closed_value = float(np.nanmedian(closed_values))
        reopened_value = (
            float(np.nanmedian(reopened_values)) if reopened_values.size else math.nan
        )
        local_quality_indices = np.unique(
            np.concatenate((closed_indices, confirmed_post))
        )
        local_quality = quality[local_quality_indices]
        quality_score = float(np.nanmedian(local_quality)) if local_quality.size else 0.0
        timing_uncertainty_ms = max(frame_period_s * 0.5, edge_gap_s * 0.5) * 1000.0
        measurable = not reasons
        possible.append(
            VisualReleaseEstimate(
                anchor_time_s=float(anchor_time_s),
                contact_time_s=contact_time,
                candidate_time_s=release_time,
                release_time_s=release_time if measurable else None,
                closure_duration_ms=closure_s * 1000.0,
                timing_uncertainty_ms=timing_uncertainty_ms,
                closed_median_aperture=closed_value,
                reopened_median_aperture=(
                    reopened_value if np.isfinite(reopened_value) else None
                ),
                valid_frame_fraction=valid_fraction,
                quality_score=quality_score,
                measurable=measurable,
                exclusion_reasons=tuple(dict.fromkeys(reasons)),
                closed_frame_times_s=tuple(float(times[i]) for i in closed_confirmed),
                reopened_frame_times_s=tuple(float(times[i]) for i in confirmed_post),
            )
        )
        index += 1

    if possible:
        # Prefer a measurable sequence, then the one nearest the independent
        # audio anchor.  No visual value is used to choose the audio event.
        return min(
            possible,
            key=lambda item: (
                not item.measurable,
                abs(float(item.candidate_time_s) - anchor_time_s),
                -(item.quality_score or 0.0),
            ),
        )

    reasons = []
    if valid_fraction < 0.60:
        reasons.append("insufficient_valid_face_frames")
    if np.count_nonzero(closed) < config.minimum_confirmation_frames:
        reasons.append("no_confirmed_lip_contact")
    if np.count_nonzero(reopened) < config.minimum_confirmation_frames * 2:
        reasons.append("insufficient_open_mouth_context")
    if not reasons:
        reasons.append("no_open_close_open_sequence")
    return VisualReleaseEstimate(
        anchor_time_s=float(anchor_time_s),
        contact_time_s=None,
        candidate_time_s=None,
        release_time_s=None,
        closure_duration_ms=None,
        timing_uncertainty_ms=None,
        closed_median_aperture=None,
        reopened_median_aperture=None,
        valid_frame_fraction=valid_fraction,
        quality_score=None,
        measurable=False,
        exclusion_reasons=tuple(reasons),
    )


def _frame_signature(
    rgb: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    if roi is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0 = int(math.floor(roi[0] * width))
        y0 = int(math.floor(roi[1] * height))
        x1 = int(math.ceil(roi[2] * width))
        y1 = int(math.ceil(roi[3] * height))
    crop = rgb[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    if not crop.size:
        return np.empty(0, dtype=np.float32)
    ys = np.linspace(0, crop.shape[0] - 1, num=min(36, crop.shape[0]), dtype=int)
    xs = np.linspace(0, crop.shape[1] - 1, num=min(64, crop.shape[1]), dtype=int)
    sampled = crop[np.ix_(ys, xs)].astype(np.float32) / 255.0
    return (
        0.2126 * sampled[..., 0]
        + 0.7152 * sampled[..., 1]
        + 0.0722 * sampled[..., 2]
    ).reshape(-1)


def extract_native_lip_samples(
    path: PathLike,
    model_path: PathLike,
    *,
    start_s: float,
    end_s: float,
    roi: tuple[float, float, float, float] | None = None,
    max_faces: int = 1,
    native_sampling_ceiling_hz: float = 1_000.0,
    repeated_frame_distance: float = 0.0015,
) -> list[LipApertureSample]:
    """Run FaceMesh on every decoded frame in a short event window.

    ``iter_sampled_video_frames`` is given a ceiling far above ordinary video
    frame rates, so each unique decoded PTS is emitted once.  Actual source PTS
    are stored in every output sample; frame-count-derived timestamps are never
    substituted.
    """

    probe = probe_media(path)
    if probe.average_fps >= native_sampling_ceiling_hz * 0.95:
        raise ValueError("native_sampling_ceiling_hz is not above the media frame rate")
    frames = list(
        iter_sampled_video_frames(
            path,
            start_s=start_s,
            end_s=end_s,
            fps=native_sampling_ceiling_hz,
        )
    )
    # On macOS, PyAV and MediaPipe may bundle incompatible FFmpeg libraries.
    # The worker keeps decoding here and FaceMesh in a subprocess that never
    # imports PyAV. A native MediaPipe failure is contained and reported as a
    # hard processing error; the analyzer's synthetic preflight normally finds
    # an unavailable sandbox graph service before any case media is scanned.
    with tempfile.TemporaryDirectory(prefix="plosive_face_worker_") as temporary:
        tracks, _, _ = analyze_face_frames_isolated(
            frames,
            model_path=Path(model_path).expanduser().resolve(),
            max_faces=max_faces,
            roi=roi,
            output_dir=Path(temporary),
            estimated_frames=len(frames),
        )
    faces_by_time: dict[float, list[tuple[int, Any]]] = {}
    for track in tracks:
        for face in track.samples:
            faces_by_time.setdefault(round(face.time_s, 9), []).append(
                (track.track_id, face)
            )

    previous_signature: np.ndarray | None = None
    output: list[LipApertureSample] = []
    for frame in frames:
        signature = _frame_signature(frame.rgb, roi)
        if (
            previous_signature is not None
            and previous_signature.shape == signature.shape
            and signature.size
        ):
            repeat_distance = float(np.mean(np.abs(signature - previous_signature)))
            repeated = repeat_distance <= repeated_frame_distance
        else:
            repeat_distance = None
            repeated = False
        previous_signature = signature

        assignments = faces_by_time.get(round(frame.time_s, 9), [])
        if assignments:
            track_id, face = max(
                assignments,
                key=lambda item: (
                    item[1].quality,
                    (item[1].bbox[2] - item[1].bbox[0])
                    * (item[1].bbox[3] - item[1].bbox[1]),
                ),
            )
            pair_values = np.asarray(face.central_lip_apertures, dtype=np.float64)
            if pair_values.size == 3 and np.all(np.isfinite(pair_values)):
                median_value = float(np.median(pair_values))
                maximum_value = float(np.max(pair_values))
                spread_value = float(np.ptp(pair_values))
            else:
                median_value = maximum_value = spread_value = math.nan
            output.append(
                LipApertureSample(
                    time_s=frame.time_s,
                    median_aperture=median_value,
                    maximum_aperture=maximum_value,
                    aperture_spread=spread_value,
                    mouth_width_px=face.mouth_width_px,
                    face_quality=face.quality,
                    pair_apertures=tuple(float(value) for value in pair_values),
                    repeated_frame=repeated,
                    repeat_distance=repeat_distance,
                    face_detected=True,
                    track_id=track_id,
                    source_pts=frame.source_pts,
                    source_time_base=frame.source_time_base,
                )
            )
        else:
            output.append(
                LipApertureSample(
                    time_s=frame.time_s,
                    median_aperture=math.nan,
                    maximum_aperture=math.nan,
                    aperture_spread=math.nan,
                    mouth_width_px=math.nan,
                    face_quality=0.0,
                    repeated_frame=repeated,
                    repeat_distance=repeat_distance,
                    face_detected=False,
                    source_pts=frame.source_pts,
                    source_time_base=frame.source_time_base,
                )
            )
    return output


@dataclass(frozen=True)
class PlosiveSyncMeasurement:
    acoustic: AcousticReleaseEstimate
    visual: VisualReleaseEstimate
    mouth_shape_at_acoustic_release: MouthShapeAtAcousticRelease | None
    lag_ms: float | None
    combined_timing_uncertainty_ms: float | None
    measurable: bool
    exclusion_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_release_estimates(
    acoustic: AcousticReleaseEstimate,
    visual: VisualReleaseEstimate,
    lip_samples: Sequence[LipApertureSample] | None = None,
    *,
    visual_config: VisualReleaseConfig | None = None,
) -> PlosiveSyncMeasurement:
    """Combine independently measured releases; positive lag means video late."""

    measurable = (
        acoustic.measurable
        and visual.measurable
        and acoustic.release_time_s is not None
        and visual.release_time_s is not None
    )
    reasons = tuple(
        dict.fromkeys((*acoustic.exclusion_reasons, *visual.exclusion_reasons))
    )
    if measurable:
        lag_ms = (visual.release_time_s - acoustic.release_time_s) * 1000.0
        visual_uncertainty = visual.timing_uncertainty_ms or 0.0
        combined_uncertainty = math.hypot(
            acoustic.time_resolution_ms * 0.5, visual_uncertainty
        )
    else:
        lag_ms = None
        combined_uncertainty = None
    mouth_shape = (
        identify_mouth_shape_at_acoustic_release(
            lip_samples, acoustic, config=visual_config
        )
        if lip_samples is not None
        else None
    )
    return PlosiveSyncMeasurement(
        acoustic=acoustic,
        visual=visual,
        mouth_shape_at_acoustic_release=mouth_shape,
        lag_ms=lag_ms,
        combined_timing_uncertainty_ms=combined_uncertainty,
        measurable=bool(measurable),
        exclusion_reasons=reasons,
    )


def lip_samples_as_records(
    samples: Iterable[LipApertureSample],
) -> list[dict[str, Any]]:
    """Return JSON-serializable raw FaceMesh event measurements."""

    return [sample.as_dict() for sample in samples]
