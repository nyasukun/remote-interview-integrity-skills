#!/usr/bin/env python3
"""Validate and unblind fixed acoustic annotations into the analysis JSON schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


STATUSES = {"measurable", "unmeasurable"}
CONFIDENCE = {"high", "medium", "low"}
REALIZATIONS = {"p", "b", "other", "uncertain"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("time must be finite")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [path.expanduser().resolve() for path in (args.events, args.blind_key, args.annotations)]
    events_path, key_path, annotations_path = paths
    if any(not path.is_file() for path in paths):
        raise SystemExit("events, blind key and annotations must exist")

    source = json.loads(events_path.read_text(encoding="utf-8"))
    source_rows = source.get("events")
    if not isinstance(source_rows, list):
        raise SystemExit("source events JSON must contain events")
    source_by_id = {}
    for event in source_rows:
        selection = event.get("selection", {}) if isinstance(event, dict) else {}
        event_id = str(selection.get("event_id") or "")
        if not event_id or event_id in source_by_id:
            raise SystemExit("source events have missing or duplicate selection.event_id")
        source_by_id[event_id] = event

    key = json.loads(key_path.read_text(encoding="utf-8"))
    key_rows = key.get("events")
    if not isinstance(key_rows, list) or not key_rows:
        raise SystemExit("blind key must contain a nonempty events list")
    key_by_blind = {str(row["blind_id"]): row for row in key_rows}
    if len(key_by_blind) != len(key_rows):
        raise SystemExit("duplicate blind_id in key")

    with annotations_path.open(encoding="utf-8", newline="") as handle:
        annotation_rows = list(csv.DictReader(handle))
    annotation_by_blind = {str(row.get("blind_id") or ""): row for row in annotation_rows}
    if len(annotation_by_blind) != len(annotation_rows) or set(annotation_by_blind) != set(key_by_blind):
        raise SystemExit("annotations must contain every blind_id exactly once")

    output_rows = []
    status_counts: Counter[str] = Counter()
    for blind_id in sorted(key_by_blind):
        key_row = key_by_blind[blind_id]
        annotation = annotation_by_blind[blind_id]
        event_id = str(key_row["runner_event_id"])
        if event_id not in source_by_id:
            raise SystemExit(f"unknown runner_event_id in key: {event_id}")
        event = source_by_id[event_id]
        selection = event["selection"]
        status = annotation.get("status", "").strip().lower()
        confidence = annotation.get("confidence", "").strip().lower()
        realization = annotation.get("acoustic_realization", "").strip().lower()
        reason = annotation.get("reason", "").strip()
        if status not in STATUSES:
            raise SystemExit(f"{blind_id}: status must be measurable or unmeasurable")
        if confidence not in CONFIDENCE:
            raise SystemExit(f"{blind_id}: confidence must be high, medium or low")
        if realization not in REALIZATIONS:
            raise SystemExit(f"{blind_id}: acoustic_realization must be p, b, other or uncertain")
        if not reason:
            raise SystemExit(f"{blind_id}: reason is required")
        status_counts[status] += 1

        anchor = float(key_row["anchor_s"])
        rank_text = annotation.get("selected_candidate_rank", "").strip()
        relative = parse_optional_float(annotation.get("selected_release_relative_ms", ""))
        selected_rank = int(rank_text) if rank_text else None
        selected_time = None
        if status == "measurable":
            if selected_rank is None and relative is None:
                raise SystemExit(f"{blind_id}: measurable rows need candidate rank or relative time")
            if selected_rank is not None:
                candidates = key_row.get("candidates", [])
                if not 1 <= selected_rank <= len(candidates):
                    raise SystemExit(f"{blind_id}: selected_candidate_rank out of range")
                rank_time = float(candidates[selected_rank - 1]["time_s"])
                if relative is not None and abs(rank_time - (anchor + relative / 1000.0)) > 0.020:
                    raise SystemExit(f"{blind_id}: candidate rank and relative time disagree by over 20 ms")
                selected_time = rank_time
                relative = (selected_time - anchor) * 1000.0
            else:
                selected_time = anchor + float(relative) / 1000.0
        elif selected_rank is not None or relative is not None:
            raise SystemExit(f"{blind_id}: unmeasurable rows must not select a time")

        output_rows.append(
            {
                "blind_id": blind_id,
                "runner_event_id": event_id,
                "status": status,
                "selected_release_time_s": selected_time,
                "selected_release_relative_ms": relative,
                "selected_candidate_rank": selected_rank,
                "confidence": confidence,
                "reason": reason,
                "acoustic_realization": realization,
                "speaker": selection.get("speaker"),
                "group": selection.get("group"),
                "epoch_id": selection.get("epoch_id"),
                "phoneme_class": selection.get("phoneme_class"),
                "target_kana": selection.get("kana"),
                "target_word": selection.get("token_text"),
                "anchor_s": anchor,
                "top3_candidates": key_row.get("candidates", []),
            }
        )

    payload = {
        "schema_version": 1,
        "analysis_role": "audio-only annotations fixed before visual join",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "blinding": key.get("blinding"),
            "input_sha256": {
                "automated_events": sha256_file(events_path),
                "blind_key": sha256_file(key_path),
                "annotations": sha256_file(annotations_path),
            },
        },
        "summary": {"event_count": len(output_rows), "status_counts": dict(status_counts)},
        "events": output_rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
