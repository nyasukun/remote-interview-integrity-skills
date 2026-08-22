#!/usr/bin/env python3
"""Join blinded acoustic release annotations to fixed visual measurements.

The annotation file is joined only by ``runner_event_id``.  Audio release
times are never inferred from video, and existing visual release times are not
re-estimated.  Positive lag means that the visible release follows audio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from video_integrity_analyzer.plosive_stats import (  # noqa: E402
    compare_candidate_to_controls,
    summarize_epochs,
)
from video_integrity_analyzer.plosive_sync import (  # noqa: E402
    AcousticReleaseEstimate,
    LipApertureSample,
    VisualReleaseConfig,
    identify_mouth_shape_at_acoustic_release,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 1729


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    label: str
    phone_classes: tuple[str, ...]
    inferential_role: str


def analysis_definitions() -> tuple[AnalysisSpec, ...]:
    return (
        AnalysisSpec("primary_p", "Primary: /p/", ("p",), "primary"),
        AnalysisSpec("secondary_b", "Secondary: /b/", ("b",), "secondary"),
        AnalysisSpec(
            "combined_exploratory",
            "Exploratory: /p/ and /b/ pooled",
            ("p", "b"),
            "exploratory",
        ),
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nested(value: object, key: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        child = value.get(key)
        if isinstance(child, Mapping):
            return child
    return {}


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _path_fingerprint(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )


def _csv_value(value: object) -> object:
    value = _json_ready(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "" if value is None else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a top-level JSON object")
    rows = data.get("events")
    if not isinstance(rows, list):
        rows = data.get("annotations")
    if not isinstance(rows, list):
        raise ValueError(
            f"{label} must contain a top-level events or annotations list"
        )
    normalized = dict(data)
    normalized["events"] = rows
    return normalized


def _index_annotation_events(
    rows: Sequence[object],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    duplicates: set[str] = set()
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"annotation event row {row_number} must be an object")
        event_id = str(raw.get("runner_event_id") or "").strip()
        if not event_id:
            raise ValueError(f"annotation event row {row_number} lacks runner_event_id")
        if event_id in indexed:
            duplicates.add(event_id)
        indexed[event_id] = raw
    if duplicates:
        raise ValueError(
            "duplicate annotation runner_event_id: " + ", ".join(sorted(duplicates))
        )
    return indexed


def _index_automated_events(
    rows: Sequence[object],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    duplicates: set[str] = set()
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"automated event row {row_number} must be an object")
        selection = _nested(raw, "selection")
        event_id = str(selection.get("event_id") or "").strip()
        if not event_id:
            raise ValueError(f"automated event row {row_number} lacks selection.event_id")
        if event_id in indexed:
            duplicates.add(event_id)
        indexed[event_id] = raw
    if duplicates:
        raise ValueError(
            "duplicate automated selection.event_id: " + ", ".join(sorted(duplicates))
        )
    return indexed


def visual_config_from_automated(
    automated: Mapping[str, object],
) -> VisualReleaseConfig:
    protocol = _nested(_nested(automated, "configuration"), "protocol")
    raw = protocol.get("visual_release_config")
    if not isinstance(raw, Mapping):
        return VisualReleaseConfig()
    allowed = {field.name for field in fields(VisualReleaseConfig)}
    return VisualReleaseConfig(
        **{key: value for key, value in raw.items() if key in allowed}
    )


def _lip_sample_from_record(raw: Mapping[str, object]) -> LipApertureSample:
    def number(key: str) -> float:
        value = _finite_float(raw.get(key))
        return value if value is not None else math.nan

    pair_values = raw.get("pair_apertures")
    pairs = (
        tuple(float(value) for value in pair_values if _finite_float(value) is not None)
        if isinstance(pair_values, (list, tuple))
        else ()
    )
    source_pts_value = raw.get("source_pts")
    return LipApertureSample(
        time_s=number("time_s"),
        median_aperture=number("median_aperture"),
        maximum_aperture=number("maximum_aperture"),
        aperture_spread=number("aperture_spread"),
        mouth_width_px=number("mouth_width_px"),
        face_quality=number("face_quality"),
        pair_apertures=pairs,
        repeated_frame=bool(raw.get("repeated_frame", False)),
        repeat_distance=_finite_float(raw.get("repeat_distance")),
        face_detected=bool(raw.get("face_detected", True)),
        track_id=(int(raw["track_id"]) if raw.get("track_id") is not None else None),
        source_pts=(int(source_pts_value) if source_pts_value is not None else None),
        source_time_base=(
            str(raw["source_time_base"])
            if raw.get("source_time_base") is not None
            else None
        ),
    )


def mouth_shape_at_annotated_release(
    automated_event: Mapping[str, object],
    *,
    audio_release_time_s: float,
    phoneme_class: str,
    confidence: object,
    config: VisualReleaseConfig,
) -> dict[str, object]:
    raw_samples = automated_event.get("lip_samples")
    if not isinstance(raw_samples, list):
        raise ValueError("automated lip_samples must be a list")
    samples: list[LipApertureSample] = []
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            raise ValueError("automated lip_samples rows must be objects")
        sample = _lip_sample_from_record(raw)
        if math.isfinite(sample.time_s):
            samples.append(sample)
    acoustic = AcousticReleaseEstimate(
        anchor_time_s=audio_release_time_s,
        phone_class=phoneme_class,
        candidate_time_s=audio_release_time_s,
        release_time_s=audio_release_time_s,
        score=None,
        runner_up_margin=None,
        acceptance_mode="blinded_annotation",
        confidence=str(confidence or "annotated"),
        measurable=True,
        time_resolution_ms=1.0,
        exclusion_reasons=(),
        selected_candidate=None,
        candidates=(),
    )
    return identify_mouth_shape_at_acoustic_release(
        samples, acoustic, config=config
    ).as_dict()


def join_blinded_annotations(
    annotations: Mapping[str, object],
    automated: Mapping[str, object],
    *,
    visual_config: VisualReleaseConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    annotation_rows = annotations.get("events")
    automated_rows = automated.get("events")
    if not isinstance(annotation_rows, list) or not isinstance(automated_rows, list):
        raise ValueError("both inputs must contain top-level events lists")
    annotation_index = _index_annotation_events(annotation_rows)
    automated_index = _index_automated_events(automated_rows)
    unknown = sorted(set(annotation_index) - set(automated_index))
    if unknown:
        raise ValueError("unknown runner_event_id: " + ", ".join(unknown))

    status_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    audio_measurable_missing_visual: list[str] = []
    invalid_mouth_shape: list[str] = []
    for event_id, automated_event in automated_index.items():
        selection = _nested(automated_event, "selection")
        annotation = annotation_index.get(event_id)
        record: dict[str, object] = {
            "runner_event_id": event_id,
            "selection": dict(selection),
            "annotation_present": annotation is not None,
            "annotation": dict(annotation) if annotation is not None else None,
            "annotation_status": "missing_annotation",
            "audio_release_time_s": None,
            "visual_release_time_s": None,
            "lag_ms": None,
            "measurable": False,
            "mouth_shape_at_audio_release": None,
            "exclusion_reasons": ["missing_blinded_annotation"],
        }
        if annotation is None:
            records.append(record)
            continue

        status = str(annotation.get("status") or "").strip().lower()
        if not status:
            raise ValueError(f"annotation {event_id} lacks status")
        status_counts[status] += 1
        selected_class = str(selection.get("phoneme_class") or "").strip("/ ").lower()
        if selected_class not in {"p", "b"}:
            raise ValueError(f"automated event {event_id} has invalid phoneme_class")
        annotated_class_value = annotation.get("phoneme_class")
        if annotated_class_value is not None:
            annotated_class = str(annotated_class_value).strip("/ ").lower()
            if annotated_class != selected_class:
                raise ValueError(
                    f"phoneme_class mismatch for {event_id}: "
                    f"annotation={annotated_class!r}, automated={selected_class!r}"
                )
        for key in ("speaker", "group", "epoch_id"):
            if not str(selection.get(key) or "").strip():
                raise ValueError(f"automated event {event_id} lacks selection.{key}")
        if selection.get("group") not in {"candidate", "control"}:
            raise ValueError(f"automated event {event_id} has invalid selection.group")

        record.update(
            annotation_status=status,
            exclusion_reasons=[],
        )
        selected_time = _finite_float(annotation.get("selected_release_time_s"))
        if status == "measurable":
            if selected_time is None or selected_time < 0:
                raise ValueError(
                    f"measurable annotation {event_id} requires a finite, non-negative "
                    "selected_release_time_s"
                )
            visual_time = _finite_float(
                _nested(_nested(automated_event, "measurement"), "visual").get(
                    "release_time_s"
                )
            )
            shape = mouth_shape_at_annotated_release(
                automated_event,
                audio_release_time_s=selected_time,
                phoneme_class=selected_class,
                confidence=annotation.get("confidence"),
                config=visual_config,
            )
            exclusions: list[str] = []
            if visual_time is None:
                exclusions.append("missing_fixed_visual_release")
                audio_measurable_missing_visual.append(event_id)
            if not bool(shape.get("valid")):
                invalid_mouth_shape.append(event_id)
            record.update(
                audio_release_time_s=selected_time,
                visual_release_time_s=visual_time,
                lag_ms=(
                    (visual_time - selected_time) * 1000.0
                    if visual_time is not None
                    else None
                ),
                measurable=visual_time is not None,
                mouth_shape_at_audio_release=shape,
                exclusion_reasons=exclusions,
            )
        elif selected_time is not None:
            raise ValueError(
                f"non-measurable annotation {event_id} must not contain "
                "selected_release_time_s"
            )
        else:
            record["exclusion_reasons"] = [f"audio_annotation_status:{status}"]
        records.append(record)

    missing_annotations = sorted(set(automated_index) - set(annotation_index))
    audit = {
        "status": "passed",
        "join_key": "annotations.runner_event_id == automated.selection.event_id",
        "automated_event_count": len(automated_index),
        "annotation_event_count": len(annotation_index),
        "matched_annotation_count": len(annotation_index),
        "automated_without_annotation_count": len(missing_annotations),
        "automated_without_annotation_ids": missing_annotations,
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "audio_measurable_annotation_count": status_counts.get("measurable", 0),
        "combined_measurable_count": sum(record["measurable"] is True for record in records),
        "audio_measurable_missing_visual_count": len(audio_measurable_missing_visual),
        "audio_measurable_missing_visual_ids": audio_measurable_missing_visual,
        "invalid_mouth_shape_count": len(invalid_mouth_shape),
        "invalid_mouth_shape_ids": invalid_mouth_shape,
        "duplicate_annotation_id_count": 0,
        "duplicate_automated_id_count": 0,
        "unknown_annotation_id_count": 0,
    }
    return records, audit


def _stats_event(record: Mapping[str, object]) -> dict[str, object]:
    selection = _nested(record, "selection")
    return {
        "speaker": selection.get("speaker"),
        "group": selection.get("group"),
        "epoch_id": selection.get("epoch_id"),
        "release_lag_ms": record.get("lag_ms"),
        "measurable": record.get("measurable") is True,
    }


def build_signed_analyses(
    records: Sequence[Mapping[str, object]],
    *,
    fps: float,
    margin_ms: float,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    analyses: dict[str, object] = {}
    epoch_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for offset, spec in enumerate(analysis_definitions()):
        selected = [
            record
            for record in records
            if _nested(record, "selection").get("phoneme_class") in spec.phone_classes
        ]
        epochs = summarize_epochs(
            (_stats_event(record) for record in selected), margin_ms=margin_ms
        )
        comparison = compare_candidate_to_controls(
            epochs,
            fps=fps,
            margin_ms=margin_ms,
            bootstrap_iterations=bootstrap_iterations,
            permutation_iterations=permutation_iterations,
            seed=seed + offset * 100,
        )
        epoch_records = [
            {"analysis": spec.key, **epoch.as_dict()} for epoch in epochs
        ]
        comparison_record = comparison.as_dict()
        analyses[spec.key] = {
            "definition": asdict(spec),
            "counts": {
                "automated_events": len(selected),
                "annotations_present": sum(
                    record.get("annotation_present") is True for record in selected
                ),
                "audio_measurable": sum(
                    record.get("annotation_status") == "measurable"
                    for record in selected
                ),
                "combined_measurable": sum(
                    record.get("measurable") is True for record in selected
                ),
            },
            "epoch_estimates": epoch_records,
            "comparison": comparison_record,
        }
        epoch_rows.extend(epoch_records)
        comparison_rows.append(
            {
                "analysis": spec.key,
                "inferential_role": spec.inferential_role,
                "phone_classes": spec.phone_classes,
                **comparison_record,
            }
        )
    return analyses, epoch_rows, comparison_rows


def _burst_shape_category(value: object) -> str | None:
    category = str(value or "").strip().lower()
    if category == "closed/contact":
        return "closed"
    if category in {"closed", "transition", "open"}:
        return category
    return None


def _eligible_burst_shape_event(
    record: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the independently timed, quality-valid burst shape for one token."""

    if record.get("annotation_status") != "measurable":
        return None
    shape = _nested(record, "mouth_shape_at_audio_release")
    if shape.get("valid") is not True:
        return None
    category = _burst_shape_category(shape.get("category"))
    mar = _finite_float(shape.get("median_aperture"))
    if category is None or mar is None:
        raise ValueError(
            f"valid mouth shape for {record.get('runner_event_id')} lacks a "
            "recognized category or finite median_aperture"
        )
    selection = _nested(record, "selection")
    return {
        "runner_event_id": record.get("runner_event_id"),
        "speaker": str(selection.get("speaker")),
        "group": str(selection.get("group")),
        "epoch_id": str(selection.get("epoch_id")),
        "phoneme_class": str(selection.get("phoneme_class")),
        "category": category,
        "is_open": category == "open",
        "mar": mar,
        "lag_ms": _finite_float(record.get("lag_ms")),
    }


def _median_central_index_options(
    population_size: int, group_size: int
) -> Iterable[tuple[int, ...]]:
    if group_size % 2:
        return ((index,) for index in range(population_size))
    return itertools.combinations(range(population_size), 2)


def _median_order_numbers(group_size: int) -> tuple[int, ...]:
    """One-based order numbers whose values determine a sample median."""

    if group_size % 2:
        return ((group_size + 1) // 2,)
    return (group_size // 2, group_size // 2 + 1)


def _count_labelings_for_central_indices(
    population_size: int,
    candidate_size: int,
    candidate_indices: Sequence[int],
    control_indices: Sequence[int],
) -> int:
    """Count binary labelings with the supplied candidate/control medians.

    A central candidate index is constrained to be selected and a central
    control index to be unselected.  Each constraint also fixes the cumulative
    number of selected epochs at that sorted index.  The free labels in the
    gaps are counted with binomial coefficients, avoiding enumeration of every
    labeling while remaining exact even when metric values are tied.
    """

    control_size = population_size - candidate_size
    constraints: dict[int, tuple[int, int]] = {}

    def add_constraint(index: int, status: int, cumulative_selected: int) -> bool:
        constraint = (status, cumulative_selected)
        previous = constraints.get(index)
        if previous is not None and previous != constraint:
            return False
        constraints[index] = constraint
        return True

    for index, selected_order in zip(
        candidate_indices, _median_order_numbers(candidate_size), strict=True
    ):
        if not add_constraint(index, 1, selected_order):
            return 0
    for index, unselected_order in zip(
        control_indices, _median_order_numbers(control_size), strict=True
    ):
        cumulative_selected = (index + 1) - unselected_order
        if not add_constraint(index, 0, cumulative_selected):
            return 0

    ways = 1
    previous_index = -1
    previous_cumulative = 0
    for index, (status, cumulative_selected) in sorted(constraints.items()):
        gap_size = index - previous_index - 1
        gap_selected = cumulative_selected - previous_cumulative - status
        if gap_selected < 0 or gap_selected > gap_size:
            return 0
        ways *= math.comb(gap_size, gap_selected)
        previous_index = index
        previous_cumulative = cumulative_selected
    tail_size = population_size - previous_index - 1
    tail_selected = candidate_size - previous_cumulative
    if tail_selected < 0 or tail_selected > tail_size:
        return 0
    return ways * math.comb(tail_size, tail_selected)


def _exact_one_sided_median_permutation(
    candidate: np.ndarray,
    control: np.ndarray,
) -> dict[str, object]:
    """Exact epoch-label permutation for a candidate-minus-control median.

    The alternative is candidate > pooled controls.  Central order-statistic
    configurations are counted combinatorially, so the result is exact without
    naively iterating every label assignment.
    """

    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if candidate.size == 0 or control.size == 0:
        return {
            "p_value": None,
            "tail_labeling_count": 0,
            "total_labeling_count": 0,
        }
    combined = np.sort(np.concatenate((candidate, control)), kind="stable")
    candidate_size = int(candidate.size)
    population_size = int(combined.size)
    observed = float(np.median(candidate) - np.median(control))
    tolerance = 1e-12 * max(1.0, abs(observed))
    tail_count = 0
    counted = 0
    candidate_options = tuple(
        _median_central_index_options(population_size, candidate_size)
    )
    control_options = tuple(
        _median_central_index_options(
            population_size, population_size - candidate_size
        )
    )
    for candidate_indices in candidate_options:
        candidate_median = float(np.mean(combined[list(candidate_indices)]))
        for control_indices in control_options:
            ways = _count_labelings_for_central_indices(
                population_size,
                candidate_size,
                candidate_indices,
                control_indices,
            )
            if ways == 0:
                continue
            control_median = float(np.mean(combined[list(control_indices)]))
            statistic = candidate_median - control_median
            counted += ways
            if statistic >= observed - tolerance:
                tail_count += ways
    expected = math.comb(population_size, candidate_size)
    if counted != expected:
        raise RuntimeError(
            "exact permutation labeling audit failed: "
            f"counted={counted}, expected={expected}"
        )
    return {
        "p_value": float(tail_count / expected),
        "tail_labeling_count": int(tail_count),
        "total_labeling_count": int(expected),
    }


def _pairwise_probability_and_cliffs_delta(
    candidate: np.ndarray, control: np.ndarray
) -> tuple[float, float]:
    differences = candidate[:, None] - control[None, :]
    wins = float(np.count_nonzero(differences > 0.0))
    ties = float(np.count_nonzero(differences == 0.0))
    total = float(differences.size)
    probability = (wins + 0.5 * ties) / total
    return probability, 2.0 * probability - 1.0


def _cluster_bootstrap_median_difference(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Resample whole epochs within group and return a percentile CI."""

    rng = np.random.default_rng(seed)
    candidate_draws = candidate[
        rng.integers(0, candidate.size, size=(iterations, candidate.size))
    ]
    control_draws = control[
        rng.integers(0, control.size, size=(iterations, control.size))
    ]
    differences = np.median(candidate_draws, axis=1) - np.median(
        control_draws, axis=1
    )
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def _burst_metric_comparison(
    epoch_rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    candidate = np.asarray(
        [float(row[metric]) for row in epoch_rows if row.get("group") == "candidate"],
        dtype=np.float64,
    )
    control = np.asarray(
        [float(row[metric]) for row in epoch_rows if row.get("group") == "control"],
        dtype=np.float64,
    )
    base = {
        "metric": metric,
        "comparison_unit": "epoch",
        "candidate_epoch_count": int(candidate.size),
        "control_epoch_count": int(control.size),
        "candidate_median": _median(candidate),
        "control_median": _median(control),
        "median_difference_candidate_minus_control": None,
        "cliffs_delta": None,
        "probability_candidate_greater": None,
        "cluster_bootstrap_ci95": None,
        "cluster_bootstrap_unit": "epoch",
        "cluster_bootstrap_iterations": int(bootstrap_iterations),
        "exploratory_one_sided_exact_permutation_p": None,
        "exact_permutation_tail_labeling_count": 0,
        "exact_permutation_total_labeling_count": 0,
        "status": "inconclusive_missing_group",
    }
    if candidate.size == 0 or control.size == 0:
        return base
    difference = float(np.median(candidate) - np.median(control))
    probability, cliffs_delta = _pairwise_probability_and_cliffs_delta(
        candidate, control
    )
    ci95 = _cluster_bootstrap_median_difference(
        candidate,
        control,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    exact = _exact_one_sided_median_permutation(candidate, control)
    base.update(
        median_difference_candidate_minus_control=difference,
        cliffs_delta=cliffs_delta,
        probability_candidate_greater=probability,
        cluster_bootstrap_ci95=ci95,
        exploratory_one_sided_exact_permutation_p=exact["p_value"],
        exact_permutation_tail_labeling_count=exact["tail_labeling_count"],
        exact_permutation_total_labeling_count=exact["total_labeling_count"],
        status="descriptive_with_exploratory_inference",
    )
    return base


def _direction_consistency_counts(
    eligible_events: Sequence[Mapping[str, object]], *, one_frame_ms: float
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    counts["burst_shape_eligible_event_count"] = len(eligible_events)
    for event in eligible_events:
        lag = _finite_float(event.get("lag_ms"))
        if lag is None:
            counts["lag_unavailable_count"] += 1
            continue
        counts["lag_available_count"] += 1
        category = str(event["category"])
        if abs(lag) <= one_frame_ms:
            counts["within_one_frame_count"] += 1
            counts["consistent_count"] += 1
        elif lag > 0.0:
            counts["positive_beyond_one_frame_count"] += 1
            if category in {"closed", "transition"}:
                counts["positive_consistent_count"] += 1
                counts["consistent_count"] += 1
            else:
                counts["positive_inconsistent_count"] += 1
                counts["inconsistent_count"] += 1
        else:
            counts["negative_beyond_one_frame_count"] += 1
            if category == "open":
                counts["negative_consistent_count"] += 1
                counts["consistent_count"] += 1
            else:
                counts["negative_inconsistent_count"] += 1
                counts["inconsistent_count"] += 1
    keys = (
        "burst_shape_eligible_event_count",
        "lag_available_count",
        "lag_unavailable_count",
        "within_one_frame_count",
        "positive_beyond_one_frame_count",
        "positive_consistent_count",
        "positive_inconsistent_count",
        "negative_beyond_one_frame_count",
        "negative_consistent_count",
        "negative_inconsistent_count",
        "consistent_count",
        "inconsistent_count",
    )
    return {key: int(counts[key]) for key in keys}


def build_burst_shape_analysis(
    records: Sequence[Mapping[str, object]],
    *,
    fps: float,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Compare mouth state at independently annotated acoustic releases."""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    one_frame_ms = 1000.0 / fps
    speaker_names = sorted(
        {
            str(_nested(record, "selection").get("speaker"))
            for record in records
            if _nested(record, "selection").get("speaker") is not None
        },
        key=lambda value: value,
    )
    all_epoch_rows: list[dict[str, object]] = []
    all_speaker_rows: list[dict[str, object]] = []
    all_comparison_rows: list[dict[str, object]] = []
    analyses: dict[str, object] = {}
    for analysis_index, spec in enumerate(analysis_definitions()):
        selected_records = [
            record
            for record in records
            if _nested(record, "selection").get("phoneme_class") in spec.phone_classes
        ]
        audio_measurable = [
            record
            for record in selected_records
            if record.get("annotation_status") == "measurable"
        ]
        eligible_events = [
            event
            for record in selected_records
            if (event := _eligible_burst_shape_event(record)) is not None
        ]

        speaker_rows: list[dict[str, object]] = []
        for speaker in speaker_names:
            source_records = [
                record
                for record in selected_records
                if str(_nested(record, "selection").get("speaker")) == speaker
            ]
            source_events = [
                event for event in eligible_events if event["speaker"] == speaker
            ]
            categories = Counter(str(event["category"]) for event in source_events)
            mars = np.asarray(
                [float(event["mar"]) for event in source_events], dtype=np.float64
            )
            group = (
                str(_nested(source_records[0], "selection").get("group"))
                if source_records
                else "unknown"
            )
            event_count = len(source_events)
            speaker_rows.append(
                {
                    "analysis": spec.key,
                    "inferential_role": spec.inferential_role,
                    "speaker": speaker,
                    "group": group,
                    "event_count": event_count,
                    "epoch_count": len(
                        {str(event["epoch_id"]) for event in source_events}
                    ),
                    "closed_count": int(categories["closed"]),
                    "transition_count": int(categories["transition"]),
                    "open_count": int(categories["open"]),
                    "open_rate": (
                        float(categories["open"] / event_count)
                        if event_count
                        else None
                    ),
                    "median_mar": _median(mars),
                }
            )

        grouped_events: dict[tuple[str, str, str], list[Mapping[str, object]]] = (
            defaultdict(list)
        )
        for event in eligible_events:
            grouped_events[
                (str(event["speaker"]), str(event["group"]), str(event["epoch_id"]))
            ].append(event)
        epoch_rows: list[dict[str, object]] = []
        for (speaker, group, epoch_id), epoch_events in sorted(
            grouped_events.items()
        ):
            categories = Counter(str(event["category"]) for event in epoch_events)
            mars = np.asarray(
                [float(event["mar"]) for event in epoch_events], dtype=np.float64
            )
            event_count = len(epoch_events)
            epoch_rows.append(
                {
                    "analysis": spec.key,
                    "inferential_role": spec.inferential_role,
                    "speaker": speaker,
                    "group": group,
                    "epoch_id": epoch_id,
                    "event_count": event_count,
                    "closed_count": int(categories["closed"]),
                    "transition_count": int(categories["transition"]),
                    "open_count": int(categories["open"]),
                    "open_rate": float(categories["open"] / event_count),
                    "median_mar": float(np.median(mars)),
                }
            )

        comparisons: dict[str, object] = {}
        comparison_rows: list[dict[str, object]] = []
        for metric_index, metric in enumerate(("open_rate", "median_mar")):
            comparison = _burst_metric_comparison(
                epoch_rows,
                metric=metric,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + analysis_index * 100 + metric_index,
            )
            comparisons[metric] = comparison
            comparison_rows.append(
                {
                    "analysis": spec.key,
                    "inferential_role": spec.inferential_role,
                    **comparison,
                }
            )

        valid_flag_count = sum(
            _nested(record, "mouth_shape_at_audio_release").get("valid") is True
            for record in audio_measurable
        )
        analyses[spec.key] = {
            "definition": asdict(spec),
            "eligibility": {
                "automated_event_count": len(selected_records),
                "audio_measurable_event_count": len(audio_measurable),
                "shape_valid_event_count": int(valid_flag_count),
                "eligible_event_count": len(eligible_events),
                "excluded_invalid_shape_count": len(audio_measurable)
                - int(valid_flag_count),
                "eligible_epoch_count": len(epoch_rows),
            },
            "speaker_descriptives": speaker_rows,
            "epoch_descriptives": epoch_rows,
            "candidate_vs_pooled_controls": comparisons,
            "visual_release_direction_consistency_qa": {
                "status": "diagnostic_counts_only_not_an_analysis_filter",
                "one_frame_ms": one_frame_ms,
                "rules": {
                    "within_one_frame": "consistent regardless of burst shape",
                    "positive_beyond_one_frame": "closed or transition is consistent",
                    "negative_beyond_one_frame": "open is consistent",
                },
                "counts": _direction_consistency_counts(
                    eligible_events, one_frame_ms=one_frame_ms
                ),
            },
        }
        all_epoch_rows.extend(epoch_rows)
        all_speaker_rows.extend(speaker_rows)
        all_comparison_rows.extend(comparison_rows)

    result = {
        "status": "burst_shape_at_independently_annotated_audio_release",
        "analysis_unit": "epoch",
        "event_inclusion": (
            "Audio annotation status is measurable and the nearest-frame mouth "
            "shape passes VisualReleaseConfig quality gates."
        ),
        "category_mapping": {
            "closed": "closed/contact",
            "transition": "transition",
            "open": "open",
        },
        "mar_definition": "median normalized inner-lip aperture at the nearest native-PTS frame",
        "token_level_fisher_test": {
            "computed": False,
            "reason": (
                "Token-level Fisher testing would treat correlated tokens within a "
                "stream epoch as independent and is therefore excluded from inference."
            ),
        },
        "comparison_notes": [
            "Candidate is compared with pooled control epochs; source speakers remain separate in descriptive rows.",
            "Cliff's delta and candidate-greater probability are pairwise epoch summaries with ties split equally.",
            "The percentile cluster bootstrap resamples whole epochs within group.",
            "The one-sided exact permutation is exploratory because call streams are not randomized or exchangeable.",
            "Direction-consistency QA is reported as counts only and never filters the burst-shape inference.",
        ],
        "analyses": analyses,
    }
    return result, all_epoch_rows, all_speaker_rows, all_comparison_rows


def _mad(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def build_posthoc_descriptives(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    speakers = sorted(
        {
            str(_nested(record, "selection").get("speaker"))
            for record in records
            if _nested(record, "selection").get("speaker") is not None
        },
        key=lambda value: value,
    )
    all_epoch_rows: list[dict[str, object]] = []
    all_speaker_rows: list[dict[str, object]] = []
    all_difference_rows: list[dict[str, object]] = []
    by_analysis: dict[str, object] = {}
    for spec in analysis_definitions():
        selected = [
            record
            for record in records
            if _nested(record, "selection").get("phoneme_class") in spec.phone_classes
        ]
        measurable = [record for record in selected if record.get("measurable") is True]
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for record in measurable:
            selection = _nested(record, "selection")
            grouped[
                (
                    str(selection.get("speaker")),
                    str(selection.get("group")),
                    str(selection.get("epoch_id")),
                )
            ].append(abs(float(record["lag_ms"])))
        epoch_rows: list[dict[str, object]] = []
        for (speaker, group, epoch_id), values in sorted(grouped.items()):
            array = np.asarray(values, dtype=np.float64)
            epoch_rows.append(
                {
                    "analysis": spec.key,
                    "speaker": speaker,
                    "group": group,
                    "epoch_id": epoch_id,
                    "event_count": int(array.size),
                    "median_absolute_lag_ms": float(np.median(array)),
                }
            )

        candidate_abs = np.asarray(
            [
                row["median_absolute_lag_ms"]
                for row in epoch_rows
                if row["group"] == "candidate"
            ],
            dtype=np.float64,
        )
        control_abs = np.asarray(
            [
                row["median_absolute_lag_ms"]
                for row in epoch_rows
                if row["group"] == "control"
            ],
            dtype=np.float64,
        )
        candidate_median = _median(candidate_abs)
        control_median = _median(control_abs)
        difference_row = {
            "analysis": spec.key,
            "candidate_epoch_count": int(candidate_abs.size),
            "control_epoch_count": int(control_abs.size),
            "candidate_median_epoch_absolute_lag_ms": candidate_median,
            "control_median_epoch_absolute_lag_ms": control_median,
            "candidate_minus_control_ms": (
                candidate_median - control_median
                if candidate_median is not None and control_median is not None
                else None
            ),
        }

        speaker_rows: list[dict[str, object]] = []
        for speaker in speakers:
            annotated_for_speaker = [
                record
                for record in selected
                if str(_nested(record, "selection").get("speaker")) == speaker
                and record.get("annotation_present") is True
            ]
            measurable_for_speaker = [
                record
                for record in measurable
                if str(_nested(record, "selection").get("speaker")) == speaker
            ]
            signed = np.asarray(
                [float(record["lag_ms"]) for record in measurable_for_speaker],
                dtype=np.float64,
            )
            absolute = np.abs(signed)
            speaker_rows.append(
                {
                    "analysis": spec.key,
                    "speaker": speaker,
                    "group": (
                        str(_nested(annotated_for_speaker[0], "selection").get("group"))
                        if annotated_for_speaker
                        else str(
                            next(
                                (
                                    _nested(record, "selection").get("group")
                                    for record in selected
                                    if str(
                                        _nested(record, "selection").get("speaker")
                                    )
                                    == speaker
                                ),
                                "unknown",
                            )
                        )
                    ),
                    "annotated_event_count": len(annotated_for_speaker),
                    "event_count": int(signed.size),
                    "epoch_count": len(
                        {
                            str(_nested(record, "selection").get("epoch_id"))
                            for record in measurable_for_speaker
                        }
                    ),
                    "signed_median_lag_ms": _median(signed),
                    "signed_mad_lag_ms": _mad(signed),
                    "absolute_median_lag_ms": _median(absolute),
                    "absolute_mad_lag_ms": _mad(absolute),
                }
            )

        by_analysis[spec.key] = {
            "definition": asdict(spec),
            "epoch_absolute_lag_medians": epoch_rows,
            "candidate_control_absolute_difference": difference_row,
            "speaker_descriptives": speaker_rows,
        }
        all_epoch_rows.extend(epoch_rows)
        all_difference_rows.append(difference_row)
        all_speaker_rows.extend(speaker_rows)

    result = {
        "status": "post_hoc_descriptive_only",
        "separation_from_signed_analyses": (
            "Absolute-lag summaries were defined after the main signed-lag protocol; "
            "they are reported without inferential p-values or confidence claims."
        ),
        "speaker_display_policy": (
            "Each source speaker is a separate row; control speakers are not pooled "
            "inside speaker-level descriptives."
        ),
        "cautions": [
            "Absolute lag discards whether video leads or follows audio.",
            "Participant stream remains confounded with device, network, and conferencing path.",
        ],
        "analyses": by_analysis,
    }
    return result, all_epoch_rows, all_speaker_rows, all_difference_rows


def event_csv_row(record: Mapping[str, object]) -> dict[str, object]:
    selection = _nested(record, "selection")
    annotation = _nested(record, "annotation")
    shape = _nested(record, "mouth_shape_at_audio_release")
    return {
        "runner_event_id": record.get("runner_event_id"),
        "speaker": selection.get("speaker"),
        "group": selection.get("group"),
        "epoch_id": selection.get("epoch_id"),
        "phoneme_class": selection.get("phoneme_class"),
        "anchor_s": selection.get("anchor_s"),
        "annotation_present": record.get("annotation_present"),
        "annotation_status": record.get("annotation_status"),
        "annotation_confidence": annotation.get("confidence"),
        "audio_release_time_s": record.get("audio_release_time_s"),
        "visual_release_time_s": record.get("visual_release_time_s"),
        "lag_ms": record.get("lag_ms"),
        "measurable": record.get("measurable"),
        "mouth_shape_category": shape.get("category"),
        "mouth_shape_frame_time_s": shape.get("frame_time_s"),
        "mouth_shape_frame_time_error_ms": shape.get("frame_time_error_ms"),
        "mouth_shape_pair_apertures": shape.get("pair_apertures"),
        "mouth_shape_median_aperture": shape.get("median_aperture"),
        "mouth_shape_maximum_aperture": shape.get("maximum_aperture"),
        "mouth_shape_aperture_spread": shape.get("aperture_spread"),
        "mouth_shape_mouth_width_px": shape.get("mouth_width_px"),
        "mouth_shape_face_quality": shape.get("face_quality"),
        "mouth_shape_repeated_frame": shape.get("repeated_frame"),
        "mouth_shape_valid": shape.get("valid"),
        "mouth_shape_exclusion_reasons": shape.get("exclusion_reasons"),
        "exclusion_reasons": record.get("exclusion_reasons"),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    annotations_path = args.annotations_json.expanduser().resolve()
    automated_path = args.automated_events_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (annotations_path, automated_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_json_object(annotations_path, "annotations JSON")
    automated = _load_json_object(automated_path, "automated events JSON")
    visual_config = visual_config_from_automated(automated)
    records, audit = join_blinded_annotations(
        annotations, automated, visual_config=visual_config
    )
    media = _nested(automated, "media")
    fps = _finite_float(media.get("average_fps"))
    if fps is None or fps <= 0:
        raise ValueError("automated events media.average_fps must be positive")
    margin_ms = 2_000.0 / fps
    signed, epoch_rows, comparison_rows = build_signed_analyses(
        records,
        fps=fps,
        margin_ms=margin_ms,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    posthoc, absolute_epoch_rows, speaker_rows, difference_rows = (
        build_posthoc_descriptives(records)
    )
    burst_shape, burst_epoch_rows, burst_speaker_rows, burst_comparison_rows = (
        build_burst_shape_analysis(
            records,
            fps=fps,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed + 10_000,
        )
    )

    planned = (
        "analysis.json",
        "events.json",
        "events.csv",
        "epoch_estimates.csv",
        "comparisons.csv",
        "posthoc_epoch_absolute.csv",
        "posthoc_speaker_descriptives.csv",
        "posthoc_candidate_control_absolute_difference.csv",
        "burst_shape_analysis.json",
        "burst_shape_epochs.csv",
        "burst_shape_speakers.csv",
        "burst_shape_comparisons.csv",
    )
    existing = [name for name in planned if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite: " + ", ".join(existing)
        )
    created_utc = _now_utc()
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "inputs": {
            "annotations_json": _path_fingerprint(annotations_path),
            "automated_events_json": _path_fingerprint(automated_path),
        },
        "configuration": {
            "fps": fps,
            "two_frame_margin_ms": margin_ms,
            "visual_release_config": asdict(visual_config),
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "seed": args.seed,
        },
        "join_audit": audit,
        "signed_lag_analyses": signed,
        "burst_shape_analysis": burst_shape,
        "post_hoc_descriptive": posthoc,
        "limitations": [
            "Blinded audio annotations improve release completeness but do not identify the cause of A/V lag.",
            "Epochs, not phoneme events, are the comparison unit for signed analyses.",
            "The two control participant streams are pooled only for group comparison; speaker rows remain separate.",
        ],
    }
    _write_json(output_dir / "analysis.json", analysis)
    _write_json(
        output_dir / "events.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": analysis["created_utc"],
            "join_audit": audit,
            "events": records,
        },
    )
    _write_csv(output_dir / "events.csv", [event_csv_row(row) for row in records])
    _write_csv(output_dir / "epoch_estimates.csv", epoch_rows)
    _write_csv(output_dir / "comparisons.csv", comparison_rows)
    _write_csv(output_dir / "posthoc_epoch_absolute.csv", absolute_epoch_rows)
    _write_csv(output_dir / "posthoc_speaker_descriptives.csv", speaker_rows)
    _write_csv(
        output_dir / "posthoc_candidate_control_absolute_difference.csv",
        difference_rows,
    )
    _write_json(
        output_dir / "burst_shape_analysis.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "configuration": {
                "fps": fps,
                "one_frame_ms": 1000.0 / fps,
                "visual_release_config": asdict(visual_config),
                "cluster_bootstrap_iterations": args.bootstrap_iterations,
                "seed": args.seed + 10_000,
            },
            **burst_shape,
        },
    )
    _write_csv(output_dir / "burst_shape_epochs.csv", burst_epoch_rows)
    _write_csv(output_dir / "burst_shape_speakers.csv", burst_speaker_rows)
    _write_csv(
        output_dir / "burst_shape_comparisons.csv", burst_comparison_rows
    )
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-json", required=True, type=Path)
    parser.add_argument("--automated-events-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--permutation-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations <= 0 or args.permutation_iterations <= 0:
        raise SystemExit("iteration counts must be positive")
    analysis = run(args)
    audit = analysis["join_audit"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "matched_annotations": audit["matched_annotation_count"],
                "combined_measurable": audit["combined_measurable_count"],
                "missing_annotations": audit["automated_without_annotation_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
