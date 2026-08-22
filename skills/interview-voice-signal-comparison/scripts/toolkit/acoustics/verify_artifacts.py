#!/usr/bin/env python3
"""Verify acoustic and designated-centered artifact directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from designated_centered import (  # type: ignore[import-not-found]
        ANALYSIS_TYPE,
        REQUIRED_SOURCE_FALSE,
        _group_metric,
        build_payload,
        csv_text,
        method_note,
    )
    from model import METRICS, implementation_provenance  # type: ignore[import-not-found]
    from signal_features import sha256_file  # type: ignore[import-not-found]
else:
    from .designated_centered import (
        ANALYSIS_TYPE,
        REQUIRED_SOURCE_FALSE,
        _group_metric,
        build_payload,
        csv_text,
        method_note,
    )
    from .model import METRICS, implementation_provenance
    from .signal_features import sha256_file


FORBIDDEN_RESULT_KEYS = {
    "aggregate_score",
    "composite_score",
    "overall_score",
    "aggregate_rank",
    "overall_rank",
    "speaker_similarity",
    "same_speaker_probability",
    "speaker_identity_score",
    "voiceprint",
}
DESIGNATED_FALSE = (
    "aggregate_score_produced",
    "aggregate_ranking_produced",
    "speaker_similarity_produced",
    "speaker_embeddings_used",
    "voiceprints_used",
    "speaker_verification_used",
    "identity_inference_produced",
    "language_labels_inferred_from_audio",
    "nationality_inference_produced",
    "affiliation_inference_produced",
)


def _artifact_errors(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return ["artifact_manifest.artifacts"]
    for raw_name, expected in artifacts.items():
        name = str(raw_name)
        if name != Path(name).name or not name:
            errors.append(f"unsafe_artifact_name:{name}")
            continue
        path = root / name
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"invalid_artifact_hash:{name}")
        elif not path.is_file():
            errors.append(f"missing_artifact:{name}")
        elif sha256_file(path) != expected.lower():
            errors.append(f"artifact_hash:{name}")
    return errors


def _find_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_RESULT_KEYS:
                findings.append(path)
            findings.extend(_find_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return findings


def _false_boundary_errors(
    payload: Mapping[str, Any], fields: Sequence[str]
) -> list[str]:
    boundary = payload.get("boundary")
    if not isinstance(boundary, Mapping):
        return ["boundary_missing"]
    return [
        f"boundary:{field}"
        for field in fields
        if type(boundary.get(field)) is not bool or boundary.get(field) is not False
    ]


def verify_acoustic_dir(analysis_dir: Path) -> dict[str, Any]:
    root = analysis_dir.expanduser().resolve()
    errors: list[str] = []
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("artifact manifest root must be an object")
    errors.extend(_artifact_errors(root, manifest))
    payload_path = root / "acoustic_features.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("acoustic payload root must be an object")
    if payload.get("analysis_type") != "non_biometric_descriptive_acoustic_features":
        errors.append("analysis_type")
    if manifest.get("analysis_type") != payload.get("analysis_type"):
        errors.append("artifact_manifest_analysis_type")
    if manifest.get("input_manifest") != payload.get("input_manifest"):
        errors.append("artifact_manifest_input")
    method = payload.get("method")
    recorded_implementation = (
        method.get("implementation") if isinstance(method, Mapping) else None
    )
    if not isinstance(recorded_implementation, Mapping):
        errors.append("implementation_provenance_missing")
    else:
        if manifest.get("implementation") != recorded_implementation:
            errors.append("artifact_manifest_implementation")
        if recorded_implementation != implementation_provenance():
            errors.append("implementation_or_runtime_mismatch")
    errors.extend(_false_boundary_errors(payload, REQUIRED_SOURCE_FALSE))
    errors.extend(f"forbidden_key:{path}" for path in _find_forbidden_keys(payload))

    input_manifest = payload.get("input_manifest")
    if not isinstance(input_manifest, Mapping):
        errors.append("input_manifest")
    else:
        try:
            source_path = Path(str(input_manifest["path"]))
            if sha256_file(source_path) != input_manifest.get("sha256"):
                errors.append("input_manifest_hash")
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            errors.append("input_manifest_hash")

    expected_metric_ids = {definition.metric_id for definition in METRICS}
    raw_groups = payload.get("group_summaries")
    clips = payload.get("clips")
    groups: dict[str, Mapping[str, Any]] = {}
    if not isinstance(raw_groups, list) or not raw_groups:
        errors.append("group_summaries")
    else:
        for index, group in enumerate(raw_groups):
            if not isinstance(group, Mapping):
                errors.append(f"group:{index}")
                continue
            group_id = str(group.get("group_id") or "")
            if not group_id or group_id in groups:
                errors.append(f"group_id:{index}")
                continue
            groups[group_id] = group
            metrics = group.get("metrics")
            if not isinstance(metrics, Mapping) or set(metrics) != expected_metric_ids:
                errors.append(f"metric_registry:{group_id}")
                continue
            try:
                clip_count = int(group.get("clip_count", 0))
                if clip_count <= 0 or clip_count != group.get("clip_count"):
                    raise ValueError
                for metric_id in expected_metric_ids:
                    stats = _group_metric(group, metric_id)
                    if int(stats["value_count"]) > clip_count:
                        errors.append(f"value_count:{group_id}:{metric_id}")
            except (TypeError, ValueError) as error:
                errors.append(f"group_stats:{group_id}:{error}")
    if not isinstance(clips, list) or len(clips) < 2:
        errors.append("clips")
    else:
        per_group: dict[str, int] = {}
        clip_ids: set[str] = set()
        for index, clip in enumerate(clips):
            if not isinstance(clip, Mapping):
                errors.append(f"clip:{index}")
                continue
            clip_id = str(clip.get("clip_id") or "")
            group_id = str(clip.get("group_id") or "")
            if not clip_id or clip_id in clip_ids:
                errors.append(f"clip_id:{index}")
            clip_ids.add(clip_id)
            if group_id not in groups:
                errors.append(f"clip_group:{clip_id}")
            per_group[group_id] = per_group.get(group_id, 0) + 1
        for group_id, group in groups.items():
            if per_group.get(group_id) != group.get("clip_count"):
                errors.append(f"clip_count:{group_id}")

    raw_artifacts = manifest.get("artifacts")
    artifact_names = set(raw_artifacts) if isinstance(raw_artifacts, Mapping) else set()
    required_artifacts = {
        "acoustic_features.json",
        "sample_mapping.json",
        "frame_features.csv",
        "clip_summary.csv",
        "group_summary.csv",
        "comparison_overview.png",
    }
    if isinstance(clips, list):
        required_artifacts.update(
            f"{clip.get('clip_id')}_acoustic_features.png"
            for clip in clips
            if isinstance(clip, Mapping) and clip.get("clip_id")
        )
    for name in sorted(required_artifacts - artifact_names):
        errors.append(f"unlisted_required_artifact:{name}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "kind": "acoustic_features",
        "analysis_dir": str(root),
        "errors": errors,
        "clip_count": len(clips) if isinstance(clips, list) else 0,
        "group_count": len(groups),
        "metric_count": len(METRICS),
        "artifact_count": (
            len(manifest["artifacts"])
            if isinstance(manifest.get("artifacts"), Mapping)
            else 0
        ),
        "artifact_manifest_sha256": sha256_file(manifest_path),
    }


def verify_designated_dir(analysis_dir: Path) -> dict[str, Any]:
    root = analysis_dir.expanduser().resolve()
    errors: list[str] = []
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("artifact manifest root must be an object")
    errors.extend(_artifact_errors(root, manifest))
    payload_path = root / "designated_centered_comparison.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("comparison payload root must be an object")
    if payload.get("analysis_type") != ANALYSIS_TYPE:
        errors.append("analysis_type")
    if manifest.get("analysis_type") != payload.get("analysis_type"):
        errors.append("artifact_manifest_analysis_type")
    if manifest.get("input") != payload.get("input"):
        errors.append("artifact_manifest_input")
    if manifest.get("configuration") != payload.get("configuration"):
        errors.append("artifact_manifest_configuration")
    if manifest.get("implementation") != payload.get("implementation"):
        errors.append("artifact_manifest_implementation")
    required_artifacts = {
        "designated_centered_comparison.json",
        "designated_centered_comparison.csv",
        "METHOD_NOTE.md",
    }
    raw_artifacts = manifest.get("artifacts")
    artifact_names = set(raw_artifacts) if isinstance(raw_artifacts, Mapping) else set()
    for name in sorted(required_artifacts - artifact_names):
        errors.append(f"unlisted_required_artifact:{name}")
    errors.extend(_false_boundary_errors(payload, DESIGNATED_FALSE))
    errors.extend(f"forbidden_key:{path}" for path in _find_forbidden_keys(payload))

    configuration = payload.get("configuration")
    input_row = payload.get("input")
    expected_payload: Mapping[str, Any] | None = None
    if not isinstance(configuration, Mapping) or not isinstance(input_row, Mapping):
        errors.append("configuration_or_input")
    else:
        try:
            expected_payload = build_payload(
                Path(str(input_row["path"])),
                designated_group_id=str(configuration["designated_group_id"]),
                comparison_group_ids=list(configuration["comparison_group_ids"]),
                include_nearest_median=bool(
                    configuration["include_nearest_median"]
                ),
                expected_input_sha256=str(input_row["sha256"]),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append(f"recompute:{error}")
    if expected_payload is not None:
        if payload != expected_payload:
            errors.append("json_recompute_mismatch")
        if (root / "designated_centered_comparison.csv").read_text(
            encoding="utf-8"
        ) != csv_text(expected_payload):
            errors.append("csv_recompute_mismatch")
        if (root / "METHOD_NOTE.md").read_text(encoding="utf-8") != method_note(
            expected_payload
        ):
            errors.append("method_note_recompute_mismatch")

    metrics = payload.get("metrics")
    comparison_ids = (
        list(configuration.get("comparison_group_ids", []))
        if isinstance(configuration, Mapping)
        else []
    )
    nearest_enabled = (
        configuration.get("include_nearest_median") is True
        if isinstance(configuration, Mapping)
        else False
    )
    if not isinstance(metrics, list) or len(metrics) != len(METRICS):
        errors.append("metric_count")
        metrics = []
    for metric in metrics:
        if not isinstance(metric, Mapping):
            errors.append("metric_object")
            continue
        comparisons = metric.get("comparisons")
        if not isinstance(comparisons, list) or [
            row.get("group_id") for row in comparisons if isinstance(row, Mapping)
        ] != comparison_ids:
            errors.append(f"comparison_groups:{metric.get('metric_id')}")
            continue
        designated_value = metric.get("designated", {}).get("value")
        distances: list[tuple[str, float]] = []
        for row in comparisons:
            minimum = row.get("min")
            maximum = row.get("max")
            median = row.get("median")
            if designated_value is None or median is None:
                if any(
                    row.get(field) is not None
                    for field in (
                        "designated_inside_inclusive_range",
                        "signed_delta_group_median_minus_designated",
                        "absolute_median_difference_native_units",
                    )
                ):
                    errors.append(
                        f"unavailable_relation:{metric.get('metric_id')}:{row.get('group_id')}"
                    )
                continue
            designated_number = float(designated_value)
            median_number = float(median)
            expected_inside = float(minimum) <= designated_number <= float(maximum)
            expected_delta = median_number - designated_number
            if row.get("designated_inside_inclusive_range") is not expected_inside:
                errors.append(
                    f"inside:{metric.get('metric_id')}:{row.get('group_id')}"
                )
            if not math.isclose(
                float(row.get("signed_delta_group_median_minus_designated")),
                expected_delta,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"signed_delta:{metric.get('metric_id')}:{row.get('group_id')}"
                )
            if not math.isclose(
                float(row.get("absolute_median_difference_native_units")),
                abs(expected_delta),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"absolute_delta:{metric.get('metric_id')}:{row.get('group_id')}"
                )
            distances.append((str(row["group_id"]), abs(expected_delta)))
        if nearest_enabled:
            expected_nearest: set[str] = set()
            if distances:
                best = min(distance for _, distance in distances)
                tolerance = max(1e-12, abs(best) * 1e-12)
                expected_nearest = {
                    group_id
                    for group_id, distance in distances
                    if abs(distance - best) <= tolerance
                }
            actual_nearest = set(
                metric.get("nearest_group_median_for_this_metric", [])
            )
            marked = {
                str(row["group_id"])
                for row in comparisons
                if row.get("is_nearest_median_for_this_metric") is True
            }
            if actual_nearest != expected_nearest or marked != expected_nearest:
                errors.append(f"nearest:{metric.get('metric_id')}")
        elif "nearest_group_median_for_this_metric" in metric or any(
            "is_nearest_median_for_this_metric" in row for row in comparisons
        ):
            errors.append(f"unexpected_nearest:{metric.get('metric_id')}")

    with (root / "designated_centered_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_rows = len(METRICS) * len(comparison_ids)
    if len(csv_rows) != expected_rows:
        errors.append("csv_row_count")
    return {
        "status": "PASS" if not errors else "FAIL",
        "kind": "designated_centered_comparison",
        "analysis_dir": str(root),
        "errors": errors,
        "metric_count": len(metrics),
        "comparison_group_count": len(comparison_ids),
        "csv_row_count": len(csv_rows),
        "artifact_count": (
            len(manifest["artifacts"])
            if isinstance(manifest.get("artifacts"), Mapping)
            else 0
        ),
        "artifact_manifest_sha256": sha256_file(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic-dir", type=Path)
    parser.add_argument("--designated-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.acoustic_dir is None and args.designated_dir is None:
        print("error: provide --acoustic-dir and/or --designated-dir", file=sys.stderr)
        return 2
    results: list[dict[str, Any]] = []
    try:
        if args.acoustic_dir is not None:
            results.append(verify_acoustic_dir(args.acoustic_dir))
        if args.designated_dir is not None:
            results.append(verify_designated_dir(args.designated_dir))
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        results.append({"status": "FAIL", "errors": [str(error)]})
    report = {
        "status": (
            "PASS" if results and all(row["status"] == "PASS" for row in results) else "FAIL"
        ),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
