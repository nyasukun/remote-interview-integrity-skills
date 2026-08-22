#!/usr/bin/env python3
"""Fail closed on common privacy and secret leaks before publication."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


MEDIA_SUFFIXES = {
    ".aac", ".avi", ".bmp", ".flac", ".gif", ".jpeg", ".jpg", ".m4a",
    ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".png", ".srt", ".tif",
    ".tiff", ".vtt", ".wav", ".webm", ".webp",
}
SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
FORBIDDEN_NAMES = {
    ".env", "blind_key.json", "credentials.json", "face_landmarker.task",
    "secrets.json",
}
FORBIDDEN_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv",
    "artifacts", "case-data", "cases", "inputs", "media", "outputs",
    "private", "recordings", "reports", "transcripts",
}
TEXT_PATTERNS = {
    "local user path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+(?:/|\b)"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
}


def repository_files(root: Path, staged: bool) -> list[Path]:
    if staged:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        names = [line for line in result.stdout.splitlines() if line]
        return [root / name for name in names]
    return sorted(
        path for path in root.rglob("*") if ".git" not in path.relative_to(root).parts
    )


def audit(root: Path, staged: bool) -> list[str]:
    findings: list[str] = []
    for path in repository_files(root, staged):
        relative = path.relative_to(root)
        lowered_parts = {part.lower() for part in relative.parts}
        if path.is_symlink():
            findings.append(f"symlink: {relative}")
            continue
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in MEDIA_SUFFIXES | SECRET_SUFFIXES:
            findings.append(f"forbidden file type: {relative}")
        if name in FORBIDDEN_NAMES or (name.startswith(".env") and name != ".env.example"):
            findings.append(f"forbidden filename: {relative}")
        if lowered_parts & FORBIDDEN_PARTS:
            findings.append(f"private/generated directory: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"file larger than 5 MiB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-text file: {relative}")
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="scan staged files only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = repository_files(root, args.staged)
    findings = audit(root, args.staged)
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}", file=sys.stderr)
        return 1
    print(f"PASS: audited {len(files)} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
