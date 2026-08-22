#!/usr/bin/env python3
"""Build and audit a chronological evidence-video manifest from curated CSV rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CLASSES = {"closure_absent", "contact_reference", "sync_reference"}
REFERENCE_CLASSES = {"contact_reference", "sync_reference"}
TRUE_VALUES = {"1", "true", "yes", "y", "include"}
FALSE_VALUES = {"0", "false", "no", "n", "exclude"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_roi(value: str) -> list[float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError("ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return parts


def include_value(raw: str) -> bool:
    value = raw.strip().lower()
    if not value:
        return True
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"invalid include value: {raw!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mouth-roi", type=parse_roi, default=[0.38, 0.50, 0.62, 0.78])
    parser.add_argument("--title", default="破裂音と口唇閉鎖の比較")
    parser.add_argument("--subtitle", default="観測資料 — 原因判定ではありません")
    parser.add_argument("--phone-label", default="/p/", help="Display label such as /p/ or /b/")
    parser.add_argument("--allow-single-category", action="store_true")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    source_csv = args.events_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not video.is_file() or not source_csv.is_file():
        raise SystemExit("video and events CSV must exist")

    with source_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"event_id", "source_release_s", "classification", "label", "note"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit("CSV must be nonempty and contain: " + ", ".join(sorted(required)))

    included: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        event_id = row["event_id"].strip()
        if not event_id or event_id in ids:
            raise SystemExit(f"row {row_number}: missing or duplicate event_id {event_id!r}")
        ids.add(event_id)
        classification = row["classification"].strip()
        if classification not in CLASSES:
            raise SystemExit(f"row {row_number}: invalid classification {classification!r}")
        try:
            release = float(row["source_release_s"])
        except ValueError as error:
            raise SystemExit(f"row {row_number}: invalid source_release_s") from error
        if not math.isfinite(release) or release < 0:
            raise SystemExit(f"row {row_number}: source_release_s must be finite and non-negative")
        include = include_value(row.get("include", ""))
        exclusion = row.get("exclusion_reason", "").strip()
        if not include and not exclusion:
            raise SystemExit(f"row {row_number}: excluded rows require exclusion_reason")
        event_roi = parse_roi(row["mouth_roi"]) if row.get("mouth_roi", "").strip() else list(args.mouth_roi)
        record = {
            "event_id": event_id,
            "source_release_s": release,
            "classification": classification,
            "label": row["label"].strip(),
            "note": row["note"].strip(),
            "mouth_roi": event_roi,
        }
        audit.append({**record, "include": include, "exclusion_reason": exclusion})
        if include:
            included.append(record)

    included.sort(key=lambda item: (float(item["source_release_s"]), str(item["event_id"])))
    counts = {name: sum(item["classification"] == name for item in included) for name in sorted(CLASSES)}
    if not included:
        raise SystemExit("no included events")
    if not args.allow_single_category and (
        counts["closure_absent"] == 0
        or not any(counts[name] for name in REFERENCE_CLASSES)
    ):
        raise SystemExit(
            "manifest requires at least one closure_absent and one "
            "contact_reference or sync_reference"
        )

    payload = {
        "schema_version": 1,
        "source_video": str(video),
        "source_sha256": sha256_file(video),
        "title": args.title,
        "subtitle": args.subtitle,
        "phone_label": args.phone_label,
        "mouth_roi": list(args.mouth_roi),
        "selection_policy": {
            "ordering": "ascending source_release_s",
            "closure_absent": "all rows fixed as include before rendering",
            "reference": "curated contact-present controls under the predeclared rule",
            "cause_inference": "not performed",
        },
        "classification_counts": counts,
        "events": included,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = output.with_name(output.stem + "_selection_audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "source_csv": str(source_csv),
                "source_csv_sha256": sha256_file(source_csv),
                "included_count": len(included),
                "excluded_count": len(rows) - len(included),
                "rows": audit,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
