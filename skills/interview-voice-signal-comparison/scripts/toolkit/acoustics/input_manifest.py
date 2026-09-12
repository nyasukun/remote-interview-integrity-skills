"""Parse caller-supplied clip manifests independently of feature extraction.

Both compact and canonical field names normalize to immutable input specs.
Group order follows first appearance in clips; labels remain caller metadata.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    from signal_features import sha256_file  # type: ignore[import-not-found]
else:
    from .signal_features import sha256_file


ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
DEFAULT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6A3D9A",
    "#A6761D",
)


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    group_label: str
    display_color_hex: str


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    group_id: str
    path: Path
    start_s: float
    end_s: float | None
    clip_label: str
    label_origin: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InputSpec:
    manifest_path: Path
    manifest_sha256: str
    groups: tuple[GroupSpec, ...]
    clips: tuple[ClipSpec, ...]


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _color(value: Any, field: str) -> str:
    text = str(value)
    if not COLOR_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a #RRGGBB color")
    return text.upper()


def _source_path(value: Any, base_dir: Path) -> Path | None:
    if isinstance(value, str):
        return _resolve_path(value, base_dir)
    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        return _resolve_path(str(value["path"]), base_dir)
    return None


def load_manifest(path: Path) -> InputSpec:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be an object")
    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, list) or len(raw_clips) < 2:
        raise ValueError("manifest must contain at least two clips")
    common_source = _source_path(
        payload.get("source") or payload.get("source_video"), manifest_path.parent
    )

    group_metadata: dict[str, dict[str, str]] = {}
    raw_groups = payload.get("groups", [])
    if isinstance(raw_groups, Mapping):
        iterable = [
            {"group_id": group_id, **(dict(value) if isinstance(value, Mapping) else {})}
            for group_id, value in raw_groups.items()
        ]
    elif isinstance(raw_groups, list):
        iterable = raw_groups
    else:
        raise ValueError("groups must be an array or object when present")
    for index, raw_group in enumerate(iterable):
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"groups[{index}] must be an object")
        group_id = str(raw_group.get("group_id") or raw_group.get("id") or "")
        if not ID_PATTERN.fullmatch(group_id) or group_id in group_metadata:
            raise ValueError(f"invalid or duplicate group id: {group_id!r}")
        color_value = raw_group.get("display_color_hex") or raw_group.get("color")
        group_metadata[group_id] = {
            "label": str(raw_group.get("group_label") or raw_group.get("label") or group_id),
            "color": _color(color_value, f"groups[{index}].display_color_hex")
            if color_value is not None
            else "",
        }

    clips: list[ClipSpec] = []
    clip_ids: set[str] = set()
    discovered_groups: list[str] = []
    group_labels: dict[str, str] = {}
    explicit_colors: dict[str, str] = {}
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, Mapping):
            raise ValueError(f"clips[{index}] must be an object")
        clip_id = str(raw_clip.get("clip_id") or raw_clip.get("id") or "")
        if not ID_PATTERN.fullmatch(clip_id) or clip_id in clip_ids:
            raise ValueError(f"invalid or duplicate clip id: {clip_id!r}")
        clip_ids.add(clip_id)
        group_id = str(raw_clip.get("group_id") or raw_clip.get("group") or "")
        if not ID_PATTERN.fullmatch(group_id):
            raise ValueError(f"clips[{index}].group_id is invalid")
        if group_id not in discovered_groups:
            discovered_groups.append(group_id)
        registered = group_metadata.get(group_id, {})
        label = str(
            raw_clip.get("group_label") or registered.get("label") or group_id
        )
        previous_label = group_labels.setdefault(group_id, label)
        if previous_label != label:
            raise ValueError(f"group {group_id!r} has inconsistent labels")
        raw_color = (
            raw_clip.get("display_color_hex")
            or raw_clip.get("group_color_hex")
            or registered.get("color")
        )
        if raw_color:
            normalized_color = _color(raw_color, f"clips[{index}].display_color_hex")
            previous_color = explicit_colors.setdefault(group_id, normalized_color)
            if previous_color != normalized_color:
                raise ValueError(f"group {group_id!r} has inconsistent colors")
        clip_source = _source_path(raw_clip.get("source"), manifest_path.parent)
        source = clip_source or common_source
        if source is None:
            raise ValueError(f"clips[{index}] has no source path")
        if not source.is_file():
            raise FileNotFoundError(source)
        start_s = _finite(raw_clip.get("start_s", 0.0), f"clips[{index}].start_s")
        end_value = raw_clip.get("end_s")
        end_s = _finite(end_value, f"clips[{index}].end_s") if end_value is not None else None
        if start_s < 0.0 or (end_s is not None and end_s <= start_s):
            raise ValueError(f"clips[{index}] has an invalid interval")
        excluded = {
            "clip_id",
            "id",
            "group_id",
            "group",
            "group_label",
            "display_color_hex",
            "group_color_hex",
            "source",
            "start_s",
            "end_s",
            "clip_label",
            "label",
            "label_origin",
        }
        clips.append(
            ClipSpec(
                clip_id=clip_id,
                group_id=group_id,
                path=source,
                start_s=start_s,
                end_s=end_s,
                clip_label=str(raw_clip.get("clip_label") or raw_clip.get("label") or clip_id),
                label_origin=str(raw_clip.get("label_origin") or "caller-supplied manifest"),
                metadata={str(key): value for key, value in raw_clip.items() if key not in excluded},
            )
        )

    groups = tuple(
        GroupSpec(
            group_id=group_id,
            group_label=group_labels[group_id],
            display_color_hex=explicit_colors.get(
                group_id, DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
            ),
        )
        for index, group_id in enumerate(discovered_groups)
    )
    return InputSpec(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        groups=groups,
        clips=tuple(clips),
    )
