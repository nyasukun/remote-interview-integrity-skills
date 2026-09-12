"""Validate reviewed target attribution without inferring a missing realization."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Mapping


def validate_automated_source(
    annotations: Mapping[str, object], automated_path: Path
) -> dict[str, object]:
    """Bind finalized annotations to the exact automated artifact they reviewed."""
    metadata = annotations.get("metadata")
    hashes = metadata.get("input_sha256") if isinstance(metadata, Mapping) else None
    recorded = hashes.get("automated_events") if isinstance(hashes, Mapping) else None
    has_recorded = isinstance(hashes, Mapping) and "automated_events" in hashes
    digest = hashlib.sha256()
    with automated_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if has_recorded and recorded != actual:
        raise ValueError(
            "recorded automated_events SHA-256 does not match automated input; "
            "use the reviewed artifact or repeat acoustic review"
        )
    return {
        "status": "verified" if has_recorded else "legacy_missing",
        "recorded_automated_events_sha256": recorded,
        "actual_automated_events_sha256": actual,
    }


def acoustic_annotation_guard(
    annotation: Mapping[str, object], phone_class: str, release_time_s: float | None
) -> tuple[str, list[str]]:
    """Keep legacy omissions explicit; reject every explicit non-target label.

    Nested ``annotation`` is the legacy joined schema. Checking both locations
    prevents a valid top-level label from hiding a conflicting original label.
    """
    nested = annotation.get("annotation")
    sources = [annotation]
    if isinstance(nested, Mapping):
        sources.append(nested)
    labels = [
        str(source["acoustic_realization"]).strip("/ ").lower()
        for source in sources
        if "acoustic_realization" in source
    ]
    intended = phone_class.strip("/ ").lower()
    reasons = []
    if labels and (intended not in {"p", "b"} or any(label != intended for label in labels)):
        reasons.append("acoustic_realization_not_target")
    if release_time_s is None or not math.isfinite(release_time_s) or release_time_s < 0:
        reasons.append("invalid_acoustic_release_time")
    for source in sources:
        if "review_window" not in source:
            continue
        window = source["review_window"]
        try:
            if not isinstance(window, Mapping):
                raise ValueError("review_window must be an object")
            if any(isinstance(window[key], bool) or not isinstance(window[key], (int, float)) for key in ("start_s", "end_s")):
                raise ValueError("review window bounds must be numbers")
            start, end = float(window["start_s"]), float(window["end_s"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("invalid review window")
            if release_time_s is not None and not start <= release_time_s < end:
                reasons.append("acoustic_release_outside_review_window")
            if "covered_intervals" in window:
                intervals = window["covered_intervals"]
                if not isinstance(intervals, list):
                    raise ValueError("covered_intervals must be a list")
                previous_end, release_covered = start, False
                for interval in intervals:
                    if not isinstance(interval, Mapping):
                        raise ValueError("coverage interval must be an object")
                    if any(isinstance(interval[key], bool) or not isinstance(interval[key], (int, float)) for key in ("start_s", "end_s")):
                        raise ValueError("coverage interval bounds must be numbers")
                    lo, hi = float(interval["start_s"]), float(interval["end_s"])
                    if not (math.isfinite(lo) and math.isfinite(hi) and previous_end <= lo < hi <= end):
                        raise ValueError("invalid coverage interval")
                    previous_end = hi
                    if release_time_s is not None and round(lo, 9) <= round(release_time_s, 9) < round(hi, 9):
                        release_covered = True
                if not release_covered:
                    reasons.append("acoustic_release_outside_decoded_coverage")
        except (KeyError, TypeError, ValueError, OverflowError):
            reasons.append("invalid_acoustic_review_window")
    if not labels:
        attribution = "legacy_unspecified"
    elif "acoustic_realization_not_target" in reasons:
        attribution = "explicit_non_target"
    else:
        attribution = "explicit_target"
    return attribution, list(dict.fromkeys(reasons))
