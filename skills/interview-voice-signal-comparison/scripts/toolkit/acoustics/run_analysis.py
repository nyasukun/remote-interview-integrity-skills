#!/usr/bin/env python3
"""Run extraction, designated-centered comparison, and verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from designated_centered import (  # type: ignore[import-not-found]
        build_payload,
        write_outputs,
    )
    from extract_acoustic_features import (  # type: ignore[import-not-found]
        extract,
        load_manifest,
    )
    from signal_features import sha256_file  # type: ignore[import-not-found]
    from verify_artifacts import (  # type: ignore[import-not-found]
        verify_acoustic_dir,
        verify_designated_dir,
    )
else:
    from .designated_centered import build_payload, write_outputs
    from .extract_acoustic_features import extract, load_manifest
    from .signal_features import sha256_file
    from .verify_artifacts import verify_acoustic_dir, verify_designated_dir


def _preflight_groups(
    manifest_path: Path,
    designated_group_id: str,
    comparison_group_ids: Sequence[str],
) -> None:
    spec = load_manifest(manifest_path)
    counts = Counter(clip.group_id for clip in spec.clips)
    if counts.get(designated_group_id) != 1:
        raise ValueError("designated group must exist and contain exactly one clip")
    if not 1 <= len(comparison_group_ids) <= 3:
        raise ValueError("choose one to three comparison groups")
    if len(set(comparison_group_ids)) != len(comparison_group_ids):
        raise ValueError("comparison groups must be unique")
    if designated_group_id in comparison_group_ids:
        raise ValueError("designated group cannot also be a comparison group")
    missing = [
        group_id
        for group_id in comparison_group_ids
        if counts.get(group_id, 0) < 1
    ]
    if missing:
        raise ValueError(f"comparison groups are missing clips: {missing}")


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    designated_group_id: str,
    comparison_group_ids: Sequence[str],
    sample_rate: int = 48_000,
    include_nearest_median: bool = False,
) -> dict[str, Any]:
    _preflight_groups(manifest_path, designated_group_id, comparison_group_ids)
    root = output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    acoustic_dir = root / "acoustic_features"
    designated_dir = root / "designated_centered"
    manifest_resolved = manifest_path.expanduser().resolve()
    configuration = {
        "input_manifest": {
            "path": str(manifest_resolved),
            "sha256": sha256_file(manifest_resolved),
        },
        "designated_group_id": designated_group_id,
        "comparison_group_ids": list(comparison_group_ids),
        "sample_rate": sample_rate,
        "include_nearest_median": bool(include_nearest_median),
        "boundary": {
            "aggregate_score_produced": False,
            "aggregate_ranking_produced": False,
            "speaker_embeddings_used": False,
            "speaker_verification_used": False,
            "identity_inference_produced": False,
            "language_labels_inferred_from_audio": False,
            "nationality_inference_produced": False,
            "affiliation_inference_produced": False,
        },
    }
    configuration_bytes = (
        json.dumps(
            configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    configuration_path = root / "run_configuration.json"
    configuration_path.write_bytes(configuration_bytes)
    configuration_sha256 = hashlib.sha256(configuration_bytes).hexdigest()
    acoustic_payload = extract(manifest_path, acoustic_dir, sample_rate)
    acoustic_verification = verify_acoustic_dir(acoustic_dir)
    if acoustic_verification["status"] != "PASS":
        raise RuntimeError(
            f"acoustic artifact verification failed: {acoustic_verification}"
        )
    acoustic_json = acoustic_dir / "acoustic_features.json"
    payload = build_payload(
        acoustic_json,
        designated_group_id=designated_group_id,
        comparison_group_ids=comparison_group_ids,
        include_nearest_median=include_nearest_median,
        expected_input_sha256=sha256_file(acoustic_json),
    )
    write_outputs(payload, designated_dir)
    designated_verification = verify_designated_dir(designated_dir)
    if designated_verification["status"] != "PASS":
        raise RuntimeError(
            "designated-centered artifact verification failed: "
            f"{designated_verification}"
        )
    verification = {
        "acoustic": acoustic_verification,
        "designated_centered": designated_verification,
    }
    report = {
        "schema_version": 1,
        "analysis_type": "non_biometric_acoustic_workflow",
        "input_manifest": configuration["input_manifest"],
        "configuration": {
            "path": "run_configuration.json",
            "sha256": configuration_sha256,
            "values": configuration,
        },
        "implementation": acoustic_payload["method"]["implementation"],
        "artifacts": {
            "run_configuration": {
                "path": "run_configuration.json",
                "sha256": configuration_sha256,
            },
            "acoustic_manifest": {
                "path": "acoustic_features/artifact_manifest.json",
                "sha256": sha256_file(acoustic_dir / "artifact_manifest.json"),
            },
            "designated_centered_manifest": {
                "path": "designated_centered/artifact_manifest.json",
                "sha256": sha256_file(designated_dir / "artifact_manifest.json"),
            },
        },
        "verification": verification,
        "boundary": {
            "aggregate_score_produced": False,
            "aggregate_ranking_produced": False,
            "speaker_embeddings_used": False,
            "speaker_verification_used": False,
            "identity_inference_produced": False,
            "language_labels_inferred_from_audio": False,
            "nationality_inference_produced": False,
            "affiliation_inference_produced": False,
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--designated-group", required=True)
    parser.add_argument(
        "--comparison-group",
        action="append",
        required=True,
        help="repeat one to three times",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--include-nearest-median", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(
            args.manifest,
            args.output_dir,
            designated_group_id=args.designated_group,
            comparison_group_ids=args.comparison_group,
            sample_rate=args.sample_rate,
            include_nearest_median=args.include_nearest_median,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
