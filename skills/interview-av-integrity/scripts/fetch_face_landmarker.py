#!/usr/bin/env python3
"""Fetch the official Face Landmarker model and verify its pinned SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MAX_BYTES = 20 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(output: Path) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output.exists():
        actual = sha256_file(output)
        if actual == MODEL_SHA256:
            return {
                "status": "ALREADY_PRESENT",
                "path": str(output),
                "sha256": actual,
                "source": MODEL_URL,
            }
        raise FileExistsError(
            f"refusing to overwrite existing file with unexpected SHA-256: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "interview-av-integrity-model-fetch/1"},
    )
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                prefix="face_landmarker.", suffix=".part", dir=output.parent, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                digest = hashlib.sha256()
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ValueError("download exceeded the 20 MiB safety limit")
                    digest.update(block)
                    handle.write(block)
        actual = digest.hexdigest()
        if actual != MODEL_SHA256:
            raise ValueError(
                f"downloaded model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}"
            )
        os.chmod(temporary_path, 0o644)
        temporary_path.replace(output)
        temporary_path = None
        return {
            "status": "DOWNLOADED",
            "path": str(output),
            "size_bytes": total,
            "sha256": actual,
            "source": MODEL_URL,
        }
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    default_output = (
        Path(__file__).resolve().parent / "toolkit" / "models" / "face_landmarker.task"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--accept-download",
        action="store_true",
        help="confirm that the official model download and its terms were reviewed",
    )
    args = parser.parse_args()
    if not args.accept_download:
        print(
            "Refusing network access without --accept-download. Review "
            "THIRD_PARTY_NOTICES.md first.",
            file=sys.stderr,
        )
        return 2
    try:
        result = fetch(args.output)
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
