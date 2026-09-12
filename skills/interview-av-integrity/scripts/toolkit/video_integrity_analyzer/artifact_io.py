"""Shared artifact I/O for the plosive analysis runners.

The three analysis scripts (``analyze_plosive_sync.py``,
``analyze_blinded_completeness.py``, ``analyze_plosive_direction_closure.py``)
write auditable JSON and CSV bundles with the same rules:

* JSON is strict: non-finite floats become ``null`` instead of the non-standard
  ``NaN``/``Infinity`` tokens, so missing geometry is visible rather than
  silently corrupting downstream parsers.
* Files are written atomically: content goes to a same-directory temporary
  file, is fsynced, and is then renamed over the target. A failed write leaves
  the previous artifact untouched and removes the temporary file.
* CSV cells keep arrays and objects JSON-encoded so that the flat table remains
  lossless for review tools.

Only formatting and file-handling behaviour lives here. Selection rules,
timing, and statistics stay in the calling scripts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence, TextIO

import numpy as np

from .plosive_sync import VisualReleaseConfig


def now_utc() -> str:
    """Return the current UTC time as an ISO 8601 string with offset."""

    return datetime.now(timezone.utc).isoformat()


def nested(value: object, key: str) -> Mapping[str, object]:
    """Return ``value[key]`` when both are mappings, otherwise an empty mapping."""

    if isinstance(value, Mapping):
        child = value.get(key)
        if isinstance(child, Mapping):
            return child
    return {}


def finite_float(value: object) -> float | None:
    """Coerce to ``float`` and return ``None`` for missing or non-finite input."""

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_ready(value: object) -> object:
    """Convert audit structures to strict JSON without hiding missing values.

    Mappings become ``dict`` with string keys, sequences become lists, NumPy
    scalars are unwrapped, and non-finite floats become ``None``. Objects that
    expose a scalar ``item()`` accessor (for example zero-dimensional arrays
    from diagnostic extensions) are unwrapped the same way; anything else is
    returned unchanged so that ``json.dumps`` reports it.
    """

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_ready(item())
        except (TypeError, ValueError):
            pass
    return value


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Yield a UTF-8 text handle whose content replaces ``path`` on success.

    The temporary file is created next to the target (``.<name>.*.tmp``) so
    the final rename stays on one filesystem. On any exception the temporary
    file is removed and the existing target is left untouched.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text``."""

    with atomic_text_writer(path) as output:
        output.write(text)


def write_json(path: Path, value: object) -> None:
    """Atomically write strict, indented, non-ASCII-preserving JSON."""

    atomic_write_text(
        path,
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )


def csv_cell(value: object) -> str:
    """Render one CSV cell: ``None``/non-finite become empty, containers JSON."""

    value = json_ready(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    empty_placeholder_column: str | None = None,
) -> None:
    """Atomically write rows with a header made of keys in first-seen order.

    ``rows`` is iterated twice, so it must be a sequence. When there are no
    columns at all, the default output is a header-only file; passing
    ``empty_placeholder_column`` instead emits that single column with one
    blank row, which some consumers rely on to detect an intentionally empty
    table.
    """

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            name = str(key)
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    if not fieldnames and empty_placeholder_column is not None:
        fieldnames = [empty_placeholder_column]
        rows = ({empty_placeholder_column: ""},)

    with atomic_text_writer(path) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def path_fingerprint(path: Path) -> dict[str, object]:
    """Describe an input file by resolved path, size, mtime, and SHA-256."""

    resolved = Path(path).expanduser().resolve()
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


def load_events_document(
    path: Path,
    label: str,
    *,
    list_keys: Sequence[str] = ("events",),
) -> dict[str, object]:
    """Load a JSON object whose event rows live under one of ``list_keys``.

    The first key holding a list wins and is exposed as ``events`` in the
    returned shallow copy, so callers can accept legacy aliases without
    changing their own join logic.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a top-level JSON object")
    rows = None
    for key in list_keys:
        candidate = data.get(key)
        if isinstance(candidate, list):
            rows = candidate
            break
    if rows is None:
        raise ValueError(
            f"{label} must contain a top-level {' or '.join(list_keys)} list"
        )
    normalized = dict(data)
    normalized["events"] = rows
    return normalized


def visual_config_from_automated(
    automated: Mapping[str, object],
) -> VisualReleaseConfig:
    """Recover the visual gates recorded by the automated runner bundle.

    Unknown keys are ignored so that newer runner output stays readable by
    older analysis scripts; absent configuration falls back to defaults.
    """

    protocol = nested(nested(automated, "configuration"), "protocol")
    raw = protocol.get("visual_release_config")
    if not isinstance(raw, Mapping):
        return VisualReleaseConfig()
    allowed = {field.name for field in fields(VisualReleaseConfig)}
    return VisualReleaseConfig(
        **{key: value for key, value in raw.items() if key in allowed}
    )
