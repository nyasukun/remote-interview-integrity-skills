#!/usr/bin/env python3
"""Fail closed on common privacy and secret leaks before publication."""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MAX_FILE_SIZE = 5 * 1024 * 1024
REGULAR_GIT_MODES = {"100644", "100755"}
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
    "email address": re.compile(
        r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b"
    ),
    "Japanese mobile number": re.compile(
        r"(?<![0-9])0[789]0(?:[- ]?[0-9]{4}){2}(?![0-9])"
    ),
    "Google Drive URL": re.compile(r"https?://drive\.google\.com/"),
    "signed URL credential": re.compile(
        r"(?:X-Amz-(?:Credential|Signature)|[?&](?:api[_-]?key|signature|token)=)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class ApprovedSyntheticPng:
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class ApprovedSyntheticMp4:
    sha256: str
    size_bytes: int


APPROVED_SYNTHETIC_PNGS = {
    PurePosixPath(
        "skills/interview-av-integrity/assets/layout-references/"
        "av-integrity-closure-review-approved.png"
    ): ApprovedSyntheticPng(
        sha256="a847a47025555f72ae83046de1169f0022105f4ef10c9028a52757200020b84c",
        width=1672,
        height=941,
    ),
    PurePosixPath(
        "skills/interview-voice-signal-comparison/assets/layout-references/"
        "voice-signal-designated-comparison-approved.png"
    ): ApprovedSyntheticPng(
        sha256="6d85aebbdefa38d886f8f78326e4ebb94b1aa66af797a9a0a4ac03c45311039a",
        width=1672,
        height=941,
    ),
}


APPROVED_SYNTHETIC_MP4S = {
    PurePosixPath(
        "examples/synthetic-interview/deterministic_synthetic_interview_blind.mp4"
    ): ApprovedSyntheticMp4(
        sha256="579bc0197f8bb03c0dc1d4250437d2fb7a77c074ec3829d7374718fe2055729c",
        size_bytes=27549459,
    ),
    PurePosixPath(
        "examples/synthetic-interview/closure_evidence_video.mp4"
    ): ApprovedSyntheticMp4(
        sha256="5275178ca4dd408b055f4ee9842d2149febbca495ce59d85acee42127173de86",
        size_bytes=21625946,
    ),
    PurePosixPath(
        "examples/synthetic-interview/voice_signal_comparison.mp4"
    ): ApprovedSyntheticMp4(
        sha256="2fd5b3305947bc043e214ce6c2b6f8dc0bb7ca543e6a2d1a8a3fc761a580f3f4",
        size_bytes=1337296,
    ),
}


@dataclass(frozen=True)
class RepositoryEntry:
    relative: PurePosixPath
    mode: str
    size: int
    data: bytes | None
    read_error: bool = False


class AuditInputError(RuntimeError):
    """Raised when the repository or index cannot be inspected safely."""


def _should_load_data(relative: PurePosixPath, size: int) -> bool:
    approved_mp4 = APPROVED_SYNTHETIC_MP4S.get(relative)
    return size <= MAX_FILE_SIZE or (
        approved_mp4 is not None and size == approved_mp4.size_bytes
    )


def _git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        command = " ".join(("git", *args))
        raise AuditInputError(f"Git command failed: {command}")
    return result.stdout


def _staged_names(root: Path) -> list[str]:
    raw = _git_bytes(
        root,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--diff-filter=ACMRT",
        "--",
    )
    try:
        names = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        raise AuditInputError("A staged path is not valid UTF-8") from exc
    return sorted(names)


def _staged_entry(root: Path, name: str) -> RepositoryEntry:
    raw = _git_bytes(root, "ls-files", "--stage", "-z", "--", name)
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise AuditInputError(f"Cannot resolve one stage-0 index entry: {name}")
    metadata, raw_path = records[0].split(b"\t", 1)
    try:
        mode, object_id, stage_number = metadata.decode("ascii").split()
        indexed_name = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise AuditInputError(f"Cannot parse the index entry: {name}") from exc
    if stage_number != "0" or indexed_name != name:
        raise AuditInputError(f"Index entry is not an unambiguous stage-0 path: {name}")

    try:
        size = int(_git_bytes(root, "cat-file", "-s", object_id).strip())
    except ValueError as exc:
        raise AuditInputError(f"Cannot determine the staged object size: {name}") from exc

    relative = PurePosixPath(name)
    data = None
    if mode in REGULAR_GIT_MODES and _should_load_data(relative, size):
        data = _git_bytes(root, "cat-file", "blob", object_id)
        if len(data) != size:
            raise AuditInputError(f"Staged object size changed during audit: {name}")
    return RepositoryEntry(relative, mode, size, data)


def _working_tree_entries(root: Path) -> list[RepositoryEntry]:
    entries: list[RepositoryEntry] = []
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if ".git" in relative_path.parts:
            continue
        try:
            metadata = path.lstat()
        except OSError:
            entries.append(
                RepositoryEntry(
                    PurePosixPath(relative_path.as_posix()),
                    "unreadable",
                    0,
                    None,
                    read_error=True,
                )
            )
            continue
        if stat.S_ISDIR(metadata.st_mode):
            continue
        relative = PurePosixPath(relative_path.as_posix())
        if stat.S_ISLNK(metadata.st_mode):
            entries.append(RepositoryEntry(relative, "120000", metadata.st_size, None))
            continue
        if not stat.S_ISREG(metadata.st_mode):
            entries.append(RepositoryEntry(relative, "unsupported", metadata.st_size, None))
            continue
        mode = "100755" if metadata.st_mode & stat.S_IXUSR else "100644"
        if not _should_load_data(relative, metadata.st_size):
            entries.append(RepositoryEntry(relative, mode, metadata.st_size, None))
            continue
        try:
            data = path.read_bytes()
        except OSError:
            entries.append(
                RepositoryEntry(relative, mode, metadata.st_size, None, read_error=True)
            )
            continue
        if len(data) != metadata.st_size:
            entries.append(
                RepositoryEntry(relative, mode, len(data), None, read_error=True)
            )
            continue
        entries.append(RepositoryEntry(relative, mode, metadata.st_size, data))
    return entries


def repository_entries(root: Path, staged: bool) -> list[RepositoryEntry]:
    if staged:
        return [_staged_entry(root, name) for name in _staged_names(root)]
    return _working_tree_entries(root)


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    if data[8:12] != b"\x00\x00\x00\r" or data[12:16] != b"IHDR":
        raise ValueError("IHDR is not the first 13-byte PNG chunk")
    return struct.unpack(">II", data[16:24])


def _audit_approved_png(
    entry: RepositoryEntry,
    approved: ApprovedSyntheticPng,
) -> list[str]:
    relative = entry.relative
    findings: list[str] = []
    if entry.mode != "100644":
        findings.append(
            f"approved PNG mode mismatch: {relative} (expected 100644)"
        )
    if entry.data is None:
        findings.append(f"approved PNG cannot be validated: {relative}")
        return findings
    digest = hashlib.sha256(entry.data).hexdigest()
    if digest != approved.sha256:
        findings.append(f"approved PNG SHA-256 mismatch: {relative}")
    try:
        dimensions = _png_dimensions(entry.data)
    except ValueError:
        findings.append(f"approved PNG structure is invalid: {relative}")
    else:
        expected = (approved.width, approved.height)
        if dimensions != expected:
            findings.append(
                "approved PNG dimensions mismatch: "
                f"{relative} (expected {approved.width}x{approved.height})"
            )
    return findings


def _audit_approved_mp4(
    entry: RepositoryEntry,
    approved: ApprovedSyntheticMp4,
) -> list[str]:
    relative = entry.relative
    findings: list[str] = []
    if entry.mode != "100644":
        findings.append(
            f"approved MP4 mode mismatch: {relative} (expected 100644)"
        )
    if entry.size != approved.size_bytes:
        findings.append(
            "approved MP4 size mismatch: "
            f"{relative} (expected {approved.size_bytes}, got {entry.size})"
        )
    if entry.data is None:
        findings.append(f"approved MP4 cannot be validated: {relative}")
        return findings
    if len(entry.data) != approved.size_bytes:
        findings.append(
            "approved MP4 data length mismatch: "
            f"{relative} (expected {approved.size_bytes}, got {len(entry.data)})"
        )
    if len(entry.data) < 8 or entry.data[4:8] != b"ftyp":
        findings.append(f"approved MP4 structure is invalid: {relative}")
    digest = hashlib.sha256(entry.data).hexdigest()
    if digest != approved.sha256:
        findings.append(f"approved MP4 SHA-256 mismatch: {relative}")
    return findings


def audit_entries(entries: list[RepositoryEntry]) -> list[str]:
    findings: list[str] = []
    for entry in entries:
        relative = entry.relative
        lowered_parts = {part.lower() for part in relative.parts}
        if relative.is_absolute() or ".." in relative.parts:
            findings.append(f"unsafe path: {relative}")
            continue
        if entry.mode == "120000":
            findings.append(f"symlink: {relative}")
            continue
        if entry.mode not in REGULAR_GIT_MODES:
            findings.append(f"unsupported file mode: {relative} ({entry.mode})")
            continue
        if entry.read_error:
            findings.append(f"unreadable file: {relative}")
            continue

        name = relative.name.lower()
        if name in FORBIDDEN_NAMES or (name.startswith(".env") and name != ".env.example"):
            findings.append(f"forbidden filename: {relative}")
        if lowered_parts & FORBIDDEN_PARTS:
            findings.append(f"private/generated directory: {relative}")

        approved_mp4 = APPROVED_SYNTHETIC_MP4S.get(relative)
        if approved_mp4 is not None:
            findings.extend(_audit_approved_mp4(entry, approved_mp4))
            continue

        if entry.size > MAX_FILE_SIZE:
            findings.append(f"file larger than 5 MiB: {relative}")

        approved_png = APPROVED_SYNTHETIC_PNGS.get(relative)
        if approved_png is not None:
            findings.extend(_audit_approved_png(entry, approved_png))
            continue

        if relative.suffix.lower() in MEDIA_SUFFIXES | SECRET_SUFFIXES:
            findings.append(f"forbidden file type: {relative}")
        if entry.data is None:
            continue
        try:
            text = entry.data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-text file: {relative}")
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="scan staged index blobs only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        entries = repository_entries(root, args.staged)
        findings = audit_entries(entries)
    except AuditInputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}", file=sys.stderr)
        return 1
    source = "staged index entries" if args.staged else "working-tree paths"
    print(f"PASS: audited {len(entries)} {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
