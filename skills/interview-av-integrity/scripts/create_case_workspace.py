#!/usr/bin/env python3
"""Create a local, non-destructive workspace for one A/V integrity case."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DIRECTORIES = (
    "protocol",
    "transcript",
    "speakers",
    "automated",
    "reviews/acoustic",
    "reviews/visual",
    "evidence",
    "qa",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    case_dir = args.case_dir.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"source video not found: {video}")
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise SystemExit("analysis range must satisfy 0 <= start < end")

    case_dir.mkdir(parents=True, exist_ok=True)
    for name in DIRECTORIES:
        (case_dir / name).mkdir(parents=True, exist_ok=True)

    config_path = case_dir / "case.json"
    protocol_path = case_dir / "protocol" / "analysis_protocol.md"
    if not args.overwrite and (config_path.exists() or protocol_path.exists()):
        raise SystemExit("case files already exist; use --overwrite only after preserving prior versions")

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    stat = video.stat()
    payload = {
        "schema_version": 1,
        "created_at": now,
        "source": {
            "path": str(video),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(video),
            "handling": "read-only; source is not copied or modified",
        },
        "analysis_scope": {"start_s": args.start, "end_s": args.end},
        "status": "initialized",
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol_path.write_text(
        "# Analysis protocol\n\n"
        f"- Created: `{now}`\n"
        f"- Source SHA-256: `{payload['source']['sha256']}`\n"
        f"- Scope: `{args.start}` to `{args.end if args.end is not None else 'media end'}` seconds\n"
        "- Primary analysis: `/p/`\n"
        "- Secondary analysis: `/b/`\n"
        "- Candidate/control assignment: TODO before measurement\n"
        "- Exclusion and quality rules: TODO before measurement\n"
        "- Reference selection rule: TODO before inspecting outcomes\n\n"
        "## Method amendments\n\nNone. Preserve this section as an append-only log.\n",
        encoding="utf-8",
    )
    print(config_path)
    print(protocol_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
