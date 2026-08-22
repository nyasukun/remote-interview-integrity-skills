#!/usr/bin/env python3
"""Build a designated-centered comparison from safe acoustic summaries.

The comparison is descriptive and metric-local.  It intentionally does not
compute speaker embeddings, cross-metric distances, an overall score, or a
ranking.  Caller-supplied group labels remain metadata.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import (  # type: ignore[import-not-found]
        BOUNDARY_TEXT,
        DESIGNATED_SCHEMA_VERSION,
        METRICS,
        json_ready,
    )
    from signal_features import sha256_file  # type: ignore[import-not-found]
else:
    from .model import BOUNDARY_TEXT, DESIGNATED_SCHEMA_VERSION, METRICS, json_ready
    from .signal_features import sha256_file


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
SOURCE_ANALYSIS_TYPE = "non_biometric_descriptive_acoustic_features"
ANALYSIS_TYPE = "designated_centered_non_biometric_descriptive_comparison"
REQUIRED_SOURCE_FALSE = (
    "aggregate_score_produced",
    "aggregate_ranking_produced",
    "speaker_embeddings_used",
    "voiceprints_used",
    "speaker_verification_used",
    "same_person_score_produced",
    "cross_group_similarity_produced",
    "identity_inference_produced",
    "group_labels_inferred_from_audio",
    "language_labels_inferred_from_audio",
    "nationality_inference_produced",
    "affiliation_inference_produced",
)
COMPOSITE_LIMITATIONS = (
    "Metrics use incompatible native units and scales: seconds, Hz, logarithmic "
    "dB/dBFS, correlations, normalized peaks, and proportions.",
    "Any normalization and weighting would be an additional, outcome-changing "
    "model choice that this descriptive analysis does not justify.",
    "Some metrics are correlated, while spectral fractions are compositional; "
    "combining them can double-count signal properties.",
    "Small clip groups do not estimate population distributions.",
    "Speech content, phonetics, prosody, microphone, room, codec, gain control, "
    "noise suppression, and conferencing mix are uncontrolled confounds.",
)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _strict_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if number <= 0 or number != value:
        raise ValueError(f"{field} must be a positive integer")
    return number


def _source_groups(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_groups = source.get("group_summaries")
    if not isinstance(raw_groups, list):
        raise ValueError("source lacks group_summaries")
    groups: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"group_summaries[{index}] must be an object")
        group_id = raw_group.get("group_id")
        if not isinstance(group_id, str) or not group_id or group_id in groups:
            raise ValueError(f"invalid or duplicate group id: {group_id!r}")
        groups[group_id] = raw_group
    return groups


def _validate_source_boundary(source: Mapping[str, Any]) -> None:
    boundary = source.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("source lacks boundary metadata")
    for field in REQUIRED_SOURCE_FALSE:
        value = boundary.get(field)
        if type(value) is not bool or value is not False:
            raise ValueError(f"source boundary field {field} must be boolean false")


def _group_metric(
    group: Mapping[str, Any], metric_id: str
) -> dict[str, float | int | bool | None]:
    raw_metrics = group.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError(f"group {group.get('group_id')!r} lacks metrics")
    raw = raw_metrics.get(metric_id)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"group {group.get('group_id')!r} lacks metric {metric_id!r}"
        )
    raw_count = raw.get("value_count")
    if isinstance(raw_count, bool):
        raise ValueError(f"{group.get('group_id')}.{metric_id}.value_count is invalid")
    try:
        value_count = int(raw_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{group.get('group_id')}.{metric_id}.value_count is invalid"
        ) from error
    if value_count < 0 or value_count != raw_count:
        raise ValueError(f"{group.get('group_id')}.{metric_id}.value_count is invalid")
    values = (raw.get("min"), raw.get("max"), raw.get("median"))
    if value_count == 0:
        if any(value is not None for value in values):
            raise ValueError(
                f"{group.get('group_id')}.{metric_id} has values with zero count"
            )
        return {
            "available": False,
            "value_count": 0,
            "min": None,
            "max": None,
            "median": None,
        }
    minimum = _finite_number(values[0], f"{metric_id}.min")
    maximum = _finite_number(values[1], f"{metric_id}.max")
    median = _finite_number(values[2], f"{metric_id}.median")
    if not minimum <= median <= maximum:
        raise ValueError(f"invalid range for {group.get('group_id')}.{metric_id}")
    return {
        "available": True,
        "value_count": value_count,
        "min": minimum,
        "max": maximum,
        "median": median,
    }


def _validate_group_selection(
    groups: Mapping[str, Mapping[str, Any]],
    designated_group_id: str,
    comparison_group_ids: Sequence[str],
) -> None:
    if designated_group_id not in groups:
        raise ValueError(f"designated group not found: {designated_group_id!r}")
    if not 1 <= len(comparison_group_ids) <= 3:
        raise ValueError("choose one to three comparison groups")
    if len(set(comparison_group_ids)) != len(comparison_group_ids):
        raise ValueError("comparison groups must be unique")
    if designated_group_id in comparison_group_ids:
        raise ValueError("designated group cannot also be a comparison group")
    for group_id in comparison_group_ids:
        if group_id not in groups:
            raise ValueError(f"comparison group not found: {group_id!r}")
    designated_count = _strict_positive_int(
        groups[designated_group_id].get("clip_count"),
        f"{designated_group_id}.clip_count",
    )
    if designated_count != 1:
        raise ValueError("designated group must contain exactly one clip")
    for group_id in comparison_group_ids:
        _strict_positive_int(
            groups[group_id].get("clip_count"), f"{group_id}.clip_count"
        )


def _validate_metric_registry(groups: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {definition.metric_id for definition in METRICS}
    for group_id, group in groups.items():
        metrics = group.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != expected:
            actual = set(metrics) if isinstance(metrics, Mapping) else set()
            raise ValueError(
                f"safe metric registry mismatch for {group_id!r}: "
                f"missing={sorted(expected - actual)}, "
                f"unregistered={sorted(actual - expected)}"
            )


def _validate_metric_definitions(source: Mapping[str, Any]) -> None:
    raw_definitions = source.get("metric_definitions")
    if not isinstance(raw_definitions, list):
        raise ValueError("source lacks metric_definitions")
    definitions: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_definitions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"metric_definitions[{index}] must be an object")
        metric_id = raw.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id or metric_id in definitions:
            raise ValueError(f"invalid or duplicate metric definition: {metric_id!r}")
        definitions[metric_id] = raw
    expected = {definition.metric_id for definition in METRICS}
    if set(definitions) != expected:
        raise ValueError("source metric definition registry does not match the toolkit")
    for definition in METRICS:
        raw = definitions[definition.metric_id]
        if (
            raw.get("unit") != definition.unit
            or raw.get("scale_note") != definition.scale_note
        ):
            raise ValueError(
                f"source definition differs for {definition.metric_id!r}"
            )


def build_payload(
    source_path: Path,
    *,
    designated_group_id: str,
    comparison_group_ids: Sequence[str],
    include_nearest_median: bool = False,
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, machine-readable descriptive comparison."""

    source_path = source_path.expanduser().resolve()
    actual_sha256 = sha256_file(source_path)
    if expected_input_sha256 is not None:
        if not SHA256_PATTERN.fullmatch(expected_input_sha256):
            raise ValueError("expected input SHA-256 must contain 64 hexadecimal digits")
        if actual_sha256.lower() != expected_input_sha256.lower():
            raise ValueError(
                f"input SHA-256 mismatch: expected {expected_input_sha256}, "
                f"got {actual_sha256}"
            )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise ValueError("source root must be an object")
    if source.get("analysis_type") != SOURCE_ANALYSIS_TYPE:
        raise ValueError("unexpected source analysis_type")
    source_method = source.get("method")
    source_implementation = (
        source_method.get("implementation")
        if isinstance(source_method, Mapping)
        else None
    )
    if not isinstance(source_implementation, Mapping):
        raise ValueError("source implementation provenance is missing")
    _validate_source_boundary(source)
    _validate_metric_definitions(source)
    groups = _source_groups(source)
    _validate_group_selection(groups, designated_group_id, comparison_group_ids)
    selected = {
        group_id: groups[group_id]
        for group_id in (designated_group_id, *comparison_group_ids)
    }
    _validate_metric_registry(selected)

    designated_group = groups[designated_group_id]
    metric_rows: list[dict[str, Any]] = []
    for definition in METRICS:
        designated_stats = _group_metric(designated_group, definition.metric_id)
        designated_value = designated_stats["median"]
        if designated_stats["available"] and not (
            designated_stats["min"]
            == designated_stats["median"]
            == designated_stats["max"]
        ):
            raise ValueError(
                f"designated one-clip range differs for {definition.metric_id}"
            )

        comparisons: list[dict[str, Any]] = []
        for group_id in comparison_group_ids:
            group = groups[group_id]
            stats = _group_metric(group, definition.metric_id)
            comparable = bool(designated_stats["available"] and stats["available"])
            if comparable:
                assert designated_value is not None
                assert stats["min"] is not None
                assert stats["max"] is not None
                assert stats["median"] is not None
                signed_delta = float(stats["median"]) - float(designated_value)
                absolute_difference = abs(signed_delta)
                inside: bool | None = bool(
                    float(stats["min"])
                    <= float(designated_value)
                    <= float(stats["max"])
                )
            else:
                signed_delta = None
                absolute_difference = None
                inside = None
            comparisons.append(
                {
                    "group_id": group_id,
                    "group_label": str(group.get("group_label") or group_id),
                    "clip_count": int(group["clip_count"]),
                    "value_count": stats["value_count"],
                    "available_for_this_metric": stats["available"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "median": stats["median"],
                    "designated_inside_inclusive_range": inside,
                    "signed_delta_group_median_minus_designated": signed_delta,
                    "absolute_median_difference_native_units": absolute_difference,
                }
            )

        metric_row: dict[str, Any] = {
            "metric_id": definition.metric_id,
            "label": definition.label,
            "unit": definition.unit,
            "scale_note": definition.scale_note,
            "directionality": "descriptive only; neither larger nor smaller is better",
            "designated": {
                "group_id": designated_group_id,
                "group_label": str(
                    designated_group.get("group_label") or designated_group_id
                ),
                "clip_count": 1,
                "available_for_this_metric": designated_stats["available"],
                "value": designated_value,
            },
            "comparisons": comparisons,
        }
        if include_nearest_median:
            distances = [
                float(row["absolute_median_difference_native_units"])
                for row in comparisons
                if row["absolute_median_difference_native_units"] is not None
            ]
            if distances:
                minimum_distance: float | None = min(distances)
                tolerance = max(1e-12, abs(minimum_distance) * 1e-12)
                nearest = [
                    str(row["group_id"])
                    for row in comparisons
                    if row["absolute_median_difference_native_units"] is not None
                    and abs(
                        float(row["absolute_median_difference_native_units"])
                        - minimum_distance
                    )
                    <= tolerance
                ]
            else:
                minimum_distance = None
                nearest = []
            for row in comparisons:
                row["is_nearest_median_for_this_metric"] = (
                    row["group_id"] in nearest
                    if row["absolute_median_difference_native_units"] is not None
                    else None
                )
            metric_row["nearest_group_median_for_this_metric"] = nearest
            metric_row[
                "nearest_absolute_median_difference_native_units"
            ] = minimum_distance
        metric_rows.append(metric_row)

    return json_ready(
        {
            "schema_version": DESIGNATED_SCHEMA_VERSION,
            "analysis_type": ANALYSIS_TYPE,
            "input": {
                "path": str(source_path),
                "sha256": actual_sha256,
                "source_schema_version": source.get("schema_version"),
                "source_analysis_type": source.get("analysis_type"),
            },
            "implementation": source_implementation,
            "configuration": {
                "designated_group_id": designated_group_id,
                "comparison_group_ids": list(comparison_group_ids),
                "include_nearest_median": bool(include_nearest_median),
            },
            "boundary": {
                "statement": BOUNDARY_TEXT,
                "aggregate_score_produced": False,
                "aggregate_ranking_produced": False,
                "speaker_similarity_produced": False,
                "speaker_embeddings_used": False,
                "voiceprints_used": False,
                "speaker_verification_used": False,
                "identity_inference_produced": False,
                "language_labels_inferred_from_audio": False,
                "nationality_inference_produced": False,
                "affiliation_inference_produced": False,
            },
            "method": {
                "designated_value": (
                    "the single designated clip value; min=median=max when available"
                ),
                "comparison_summary": (
                    "unweighted min, max, and median over per-clip values"
                ),
                "range_rule": (
                    "inclusive: group_min <= designated_value <= group_max"
                ),
                "signed_delta_rule": (
                    "comparison_group_median - designated_value in this metric's "
                    "native unit"
                ),
                "nearest_rule": (
                    "optional; smallest absolute designated-to-group-median "
                    "difference independently within each metric; ties retained"
                ),
                "unavailable_rule": (
                    "null when a metric is unavailable, including stereo metrics "
                    "for mono material"
                ),
                "aggregation_across_metrics": "none",
            },
            "groups": [
                {
                    "group_id": group_id,
                    "group_label": str(
                        groups[group_id].get("group_label") or group_id
                    ),
                    "clip_count": int(groups[group_id]["clip_count"]),
                    "clip_ids": list(groups[group_id].get("clip_ids", [])),
                    "display_color_hex": groups[group_id].get("display_color_hex"),
                }
                for group_id in (designated_group_id, *comparison_group_ids)
            ],
            "metrics": metric_rows,
            "non_composability": {
                "composite_valid": False,
                "reasons": list(COMPOSITE_LIMITATIONS),
            },
        }
    )


CSV_FIELDS = (
    "metric_id",
    "label",
    "unit",
    "scale_note",
    "designated_group_id",
    "designated_value",
    "comparison_group_id",
    "comparison_group_label",
    "comparison_clip_count",
    "comparison_value_count",
    "group_min",
    "group_max",
    "group_median",
    "designated_inside_inclusive_range",
    "signed_delta_group_median_minus_designated",
    "absolute_median_difference_native_units",
    "is_nearest_median_for_this_metric",
)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if type(value) is bool:
        return str(value).lower()
    return value


def csv_text(payload: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for metric in payload["metrics"]:
        for comparison in metric["comparisons"]:
            writer.writerow(
                {
                    "metric_id": metric["metric_id"],
                    "label": metric["label"],
                    "unit": metric["unit"],
                    "scale_note": metric["scale_note"],
                    "designated_group_id": metric["designated"]["group_id"],
                    "designated_value": _csv_value(metric["designated"]["value"]),
                    "comparison_group_id": comparison["group_id"],
                    "comparison_group_label": comparison["group_label"],
                    "comparison_clip_count": comparison["clip_count"],
                    "comparison_value_count": comparison["value_count"],
                    "group_min": _csv_value(comparison["min"]),
                    "group_max": _csv_value(comparison["max"]),
                    "group_median": _csv_value(comparison["median"]),
                    "designated_inside_inclusive_range": _csv_value(
                        comparison["designated_inside_inclusive_range"]
                    ),
                    "signed_delta_group_median_minus_designated": _csv_value(
                        comparison["signed_delta_group_median_minus_designated"]
                    ),
                    "absolute_median_difference_native_units": _csv_value(
                        comparison["absolute_median_difference_native_units"]
                    ),
                    "is_nearest_median_for_this_metric": _csv_value(
                        comparison.get("is_nearest_median_for_this_metric")
                    ),
                }
            )
    return output.getvalue()


def method_note(payload: Mapping[str, Any]) -> str:
    configuration = payload["configuration"]
    input_row = payload["input"]
    designated = configuration["designated_group_id"]
    comparisons = ", ".join(
        f"`{group_id}`" for group_id in configuration["comparison_group_ids"]
    )
    nearest_text = (
        "Enabled. Any nearest-median relation is reported separately per metric; "
        "ties are retained."
        if configuration["include_nearest_median"]
        else "Disabled. No nearest-median relation was produced."
    )
    reasons = "\n".join(f"- {reason}" for reason in COMPOSITE_LIMITATIONS)
    return f"""# Designated-centered descriptive acoustic comparison

## Scope and evidence boundary

This artifact compares the caller-designated one-clip group `{designated}` with
the caller-selected comparison groups {comparisons}. It describes recorded
signals only. It does not establish a shared speaker, identity, nationality,
affiliation, language background, or intent.

- Input: `{input_row['path']}`
- Input SHA-256: `{input_row['sha256']}`
- Nearest-median option: {nearest_text}

## Calculations

The designated value is its single clip value. Each comparison group is
summarized with the unweighted minimum, maximum, and median across clips.
Inclusive containment means `group_min <= designated_value <= group_max`.
The signed delta is `comparison_group_median - designated_value` in the
metric's native unit. A null result means that the metric was unavailable,
such as a stereo diagnostic for mono material.

## Why metrics are not combined

{reasons}

Consequently, no cross-metric distance, overall score, vote, or ordering is
validly produced. Per-metric containment, delta, and optional nearest-median
relations must not be aggregated or interpreted as speaker similarity.

## Machine-readable artifacts

- `designated_centered_comparison.json`: complete definitions and values
- `designated_centered_comparison.csv`: one metric-comparison pair per row
- `artifact_manifest.json`: SHA-256 hashes for integrity checking
"""


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "designated_centered_comparison.json"
    csv_path = output_dir / "designated_centered_comparison.csv"
    note_path = output_dir / "METHOD_NOTE.md"
    json_path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    csv_path.write_text(csv_text(payload), encoding="utf-8")
    note_path.write_text(method_note(payload), encoding="utf-8")
    artifacts = {
        path.name: sha256_file(path) for path in (json_path, csv_path, note_path)
    }
    manifest = {
        "schema_version": DESIGNATED_SCHEMA_VERSION,
        "analysis_type": payload["analysis_type"],
        "input": payload["input"],
        "configuration": payload["configuration"],
        "implementation": payload["implementation"],
        "artifacts": artifacts,
        "boundary": BOUNDARY_TEXT,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--designated-group", required=True)
    parser.add_argument(
        "--comparison-group",
        action="append",
        required=True,
        help="repeat one to three times",
    )
    parser.add_argument("--include-nearest-median", action="store_true")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(
            args.input,
            designated_group_id=args.designated_group,
            comparison_group_ids=args.comparison_group,
            include_nearest_median=args.include_nearest_median,
            expected_input_sha256=args.expected_input_sha256,
        )
        write_outputs(payload, args.output_dir)
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"Wrote {len(payload['metrics'])} metric-local comparisons "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
