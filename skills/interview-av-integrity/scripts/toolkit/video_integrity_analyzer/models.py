from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MediaProbe:
    path: str
    size_bytes: int
    duration_s: float
    video_codec: str
    width: int
    height: int
    average_fps: float
    video_time_base: str
    video_start_s: float
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    audio_time_base: str | None
    audio_start_s: float | None
    container_format: str
    sha256: str | None = None


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    face_width_px: float
    mouth_width_px: float
    mouth_open: float
    mouth_texture_ratio: float
    quality: float
    normalized_lip_shape: tuple[float, ...]
    appearance_descriptor: tuple[float, ...] = ()
    # Raw, scale-normalized central inner-lip distances.  These are retained
    # separately from ``mouth_open`` so event-level analyses can use explicit
    # lip-contact rules without changing the legacy whole-speech metric.
    central_lip_apertures: tuple[float, ...] = ()
    lip_aperture_spread: float = float("nan")


@dataclass(frozen=True)
class FaceSample:
    time_s: float
    mouth_open: float
    mouth_texture_ratio: float
    quality: float
    bbox: tuple[float, float, float, float]
    normalized_lip_shape: tuple[float, ...]
    mouth_width_px: float = float("nan")
    central_lip_apertures: tuple[float, ...] = ()
    lip_aperture_spread: float = float("nan")


@dataclass
class FaceTrack:
    track_id: int
    samples: list[FaceSample] = field(default_factory=list)
    last_bbox: tuple[float, float, float, float] | None = None
    last_seen_s: float = -1.0
    appearance_descriptor: tuple[float, ...] = ()
    appearance_sample_count: int = 0


@dataclass(frozen=True)
class SyncWindow:
    start_s: float
    end_s: float
    lag_ms: float | None
    peak_corr: float | None
    prominence: float | None
    voiced_s: float
    face_valid_fraction: float
    confidence: str
    reason: str


@dataclass(frozen=True)
class VisualEvent:
    time_s: float
    indicator: str
    score: float
    explanation: str


@dataclass(frozen=True)
class LearnedSyncWindow:
    start_s: float
    end_s: float
    offset_ms: float | None
    confidence_score: float | None
    min_distance: float | None
    crop_stability_ms: float | None
    correspondence_screen_pass: bool
    status: str
    reason: str


@dataclass
class TrackAnalysis:
    track_id: int
    observed_start_s: float
    observed_end_s: float
    sample_count: int
    usable_sample_count: int
    median_lag_ms: float | None
    consistent_lag_fraction: float | None
    sync_windows: list[SyncWindow]
    suspicious_sync_windows: list[SyncWindow]
    visual_events: list[VisualEvent]
    review_priority: str
    learned_sync_windows: list[LearnedSyncWindow] = field(default_factory=list)


@dataclass
class AnalysisResult:
    tool_version: str
    created_at: str
    media: MediaProbe
    settings: dict[str, Any]
    tracks: list[TrackAnalysis]
    common_sync_issue: bool
    common_sync_explanation: str
    limitations: list[str]
    output_dir: str
    thumbnails: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisOptions:
    output_dir: Path
    face_model_path: Path
    analysis_fps: float = 10.0
    feature_rate_hz: float = 100.0
    max_faces: int = 4
    start_s: float = 0.0
    end_s: float | None = None
    sync_threshold_ms: float = 180.0
    baseline_lag_ms: float = 0.0
    window_s: float = 8.0
    stride_s: float = 2.0
    max_lag_ms: float = 600.0
    roi: tuple[float, float, float, float] | None = None
    max_thumbnails: int = 12
    save_thumbnails: bool = True
    compute_hash: bool = False
    learned_sync: bool = False
    syncnet_model_path: Path | None = None
    learned_max_windows: int = 8
