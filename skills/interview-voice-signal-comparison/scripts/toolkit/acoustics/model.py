"""Shared artifact schema for the non-biometric acoustic toolkit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ACOUSTIC_SCHEMA_VERSION = 1
DESIGNATED_SCHEMA_VERSION = 1
BOUNDARY_TEXT = (
    "Descriptive signal observations only. No aggregate score or ranking, "
    "speaker embedding, voiceprint, speaker verification, identity, nationality, "
    "or affiliation inference."
)

IMPLEMENTATION_FILES = (
    "designated_centered.py",
    "extract_acoustic_features.py",
    "input_manifest.py",
    "model.py",
    "run_analysis.py",
    "signal_features.py",
    "verify_artifacts.py",
)
RUNTIME_DISTRIBUTIONS = ("av", "matplotlib", "numpy", "Pillow", "scipy")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_provenance() -> dict[str, Any]:
    """Return portable source hashes and runtime versions for reproduction."""

    root = Path(__file__).resolve().parent
    sources = {
        name: _sha256_file(root / name)
        for name in IMPLEMENTATION_FILES
    }
    distributions: dict[str, str] = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "unavailable"
    runtime = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "distributions": distributions,
    }
    canonical = json.dumps(
        {"source_files": sources, "runtime": runtime},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "source_files": sources,
        "runtime": runtime,
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
    }


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    label: str
    unit: str
    scale_note: str
    path: tuple[str, ...]


METRICS = (
    MetricDefinition("duration_s", "Clip duration", "s", "linear time", ("duration_s",)),
    MetricDefinition(
        "active_frame_fraction",
        "Active-frame fraction",
        "ratio",
        "bounded proportion on 0..1",
        ("activity", "active_frame_fraction"),
    ),
    MetricDefinition(
        "voiced_fraction_of_active",
        "Voiced fraction of active frames",
        "ratio",
        "bounded conditional proportion on 0..1",
        ("f0_hz", "voiced_fraction_of_active"),
    ),
    MetricDefinition(
        "f0_median_hz",
        "Median voiced F0",
        "Hz",
        "frequency",
        ("f0_hz", "distribution", "median"),
    ),
    MetricDefinition(
        "f0_iqr_hz",
        "Voiced F0 IQR",
        "Hz",
        "frequency spread",
        ("f0_hz", "distribution", "iqr"),
    ),
    MetricDefinition(
        "periodicity_median",
        "Median periodicity",
        "normalized peak",
        "bounded normalized autocorrelation peak on 0..1",
        ("periodicity", "median"),
    ),
    MetricDefinition(
        "rms_dbfs",
        "Clip RMS",
        "dBFS",
        "logarithmic amplitude level",
        ("waveform", "rms_dbfs"),
    ),
    MetricDefinition(
        "spectral_centroid_median_hz",
        "Median spectral centroid",
        "Hz",
        "power-weighted frequency",
        ("spectral", "centroid_hz", "median"),
    ),
    MetricDefinition(
        "spectral_flatness_median",
        "Median spectral flatness",
        "ratio",
        "dimensionless geometric-to-arithmetic power ratio",
        ("spectral", "flatness", "median"),
    ),
    MetricDefinition(
        "stereo_balance_median_db",
        "Median L/R balance",
        "dB",
        "logarithmic channel RMS ratio",
        ("stereo", "balance_db", "median"),
    ),
    MetricDefinition(
        "stereo_correlation_median",
        "Median L/R correlation",
        "correlation",
        "bounded centered correlation on -1..1",
        ("stereo", "correlation", "median"),
    ),
    MetricDefinition(
        "stereo_side_fraction_median",
        "Median stereo side fraction",
        "ratio",
        "bounded side-energy fraction on 0..1",
        ("stereo", "side_fraction", "median"),
    ),
    MetricDefinition(
        "mean_active_band_fraction_low_0_500",
        "Mean active 0–500 Hz power fraction",
        "power fraction",
        "bounded component of a compositional spectrum",
        ("spectral", "mean_active_band_fractions", "low_0_500"),
    ),
    MetricDefinition(
        "mean_active_band_fraction_mid_500_2000",
        "Mean active 500–2,000 Hz power fraction",
        "power fraction",
        "bounded component of a compositional spectrum",
        ("spectral", "mean_active_band_fractions", "mid_500_2000"),
    ),
    MetricDefinition(
        "mean_active_band_fraction_high_2000_8000",
        "Mean active 2,000–8,000 Hz power fraction",
        "power fraction",
        "bounded component of a compositional spectrum",
        ("spectral", "mean_active_band_fractions", "high_2000_8000"),
    ),
    MetricDefinition(
        "mean_active_band_fraction_upper_8000_nyquist",
        "Mean active 8,000 Hz–Nyquist power fraction",
        "power fraction",
        "bounded component of a compositional spectrum",
        ("spectral", "mean_active_band_fractions", "upper_8000_nyquist"),
    ),
)


def nested(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def scalar_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for definition in METRICS:
        value = nested(summary, definition.path)
        if value is None:
            result[definition.metric_id] = None
            continue
        number = float(value)
        result[definition.metric_id] = number if math.isfinite(number) else None
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
