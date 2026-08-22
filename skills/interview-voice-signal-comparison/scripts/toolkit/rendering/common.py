from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 24
AUDIO_RATE = 48_000
SAMPLES_PER_FRAME = AUDIO_RATE // FPS
BACKGROUND = (5, 12, 18)
PANEL = (8, 18, 26)
TEXT = (232, 238, 243)
MUTED = (151, 164, 175)
GRID = (55, 70, 81)
DEFAULT_LIMITATION = "観測できる音声信号の違いを表示。本人性は判定しない。"
LIMITATION_DETAIL = "同一性、国籍、所属、意図、欺瞞を推定しません。"


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    role: str
    label: str
    short_label: str
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    group_id: str
    label: str
    language: str
    label_origin: str
    source: Path
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class RenderSpec:
    title: str
    limitation_text: str
    source_manifest: Path
    groups: tuple[GroupSpec, ...]
    clips: tuple[ClipSpec, ...]
    intro_s: float
    gap_s: float
    outro_s: float
    edge_fade_ms: float
    provenance: tuple[dict[str, str], ...]
    anchor_group: str
    acoustic_artifacts: dict[str, dict[str, str]]
    clip_manifest: Path
    clip_manifest_sha256: str

    @property
    def comparison_groups(self) -> tuple[GroupSpec, ...]:
        return tuple(group for group in self.groups if group.role == "comparison")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ManifestError(f"{name} must be a finite number")
    return number


def _relative_file(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise ManifestError(f"{name} does not exist: {path}")
    return path


def load_manifest(path: Path) -> RenderSpec:
    manifest = path.expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ManifestError("render manifest must be an object with schema_version 2")
    raw_clip_manifest = payload.get("clip_manifest")
    if not isinstance(raw_clip_manifest, dict):
        raise ManifestError("clip_manifest must contain path and sha256")
    clip_manifest = _relative_file(manifest.parent, raw_clip_manifest.get("path"), "clip_manifest.path")
    clip_manifest_hash = sha256(clip_manifest)
    if raw_clip_manifest.get("sha256") != clip_manifest_hash:
        raise ManifestError("clip_manifest SHA-256 mismatch")
    clip_payload = json.loads(clip_manifest.read_text(encoding="utf-8"))
    if not isinstance(clip_payload, dict) or clip_payload.get("schema_version") != 1:
        raise ManifestError("clip manifest must be an object with schema_version 1")

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ManifestError("comparison must be an object")
    anchor_group = str(comparison.get("anchor_group", "")).strip()
    display_anchor = str(comparison.get("display_anchor_group", "")).strip()
    point_group = str(comparison.get("point_group", "")).strip()
    reference_groups = comparison.get("reference_groups")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", anchor_group):
        raise ManifestError("comparison.anchor_group must be a generic group ID")
    if not anchor_group == display_anchor == point_group:
        raise ManifestError("anchor_group, display_anchor_group, and point_group must match")
    if not isinstance(reference_groups, list) or not 1 <= len(reference_groups) <= 3:
        raise ManifestError("comparison.reference_groups must contain one to three groups")
    reference_groups = [str(value) for value in reference_groups]
    if len(set(reference_groups)) != len(reference_groups) or anchor_group in reference_groups:
        raise ManifestError("comparison reference groups must be unique and exclude the anchor")
    if comparison.get("delta_definition") != "group metric minus user-designated point metric":
        raise ManifestError("comparison.delta_definition must declare group minus designated")

    raw_groups = clip_payload.get("groups")
    if not isinstance(raw_groups, list) or not 2 <= len(raw_groups) <= 4:
        raise ManifestError("clip manifest groups must contain anchor plus one to three comparison groups")
    if clip_payload.get("designated_group_id") != anchor_group:
        raise ManifestError("clip manifest designated_group_id must match the render anchor")
    if clip_payload.get("comparison_group_ids") != reference_groups:
        raise ManifestError("clip manifest comparison_group_ids must match comparison.reference_groups")
    groups: list[GroupSpec] = []
    seen_groups: set[str] = set()
    for index, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            raise ManifestError(f"groups[{index}] must be an object")
        group_id = str(item.get("group_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", group_id) or group_id in seen_groups:
            raise ManifestError(f"groups[{index}].group_id is invalid or duplicate")
        seen_groups.add(group_id)
        source_role = str(item.get("role", ""))
        expected_role = "designated_anchor" if group_id == anchor_group else "comparison"
        allowed_source_roles = {"designated_anchor", "designated_point"} if expected_role == "designated_anchor" else {"comparison", "comparison_reference"}
        if source_role not in allowed_source_roles:
            raise ManifestError(f"{group_id}: group role does not match comparison configuration")
        raw_color = item.get("color_rgb")
        hex_color = item.get("display_color_hex")
        if hex_color is not None:
            text_color = str(hex_color)
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text_color):
                raise ManifestError(f"{group_id}: display_color_hex must be #RRGGBB")
            hex_values = tuple(int(text_color[index : index + 2], 16) for index in (1, 3, 5))
            if raw_color is not None and (
                not isinstance(raw_color, list)
                or len(raw_color) != 3
                or tuple(int(value) for value in raw_color) != hex_values
            ):
                raise ManifestError(f"{group_id}: color_rgb and display_color_hex disagree")
            color = hex_values
        else:
            if not isinstance(raw_color, list) or len(raw_color) != 3:
                raise ManifestError(f"{group_id}: color_rgb or display_color_hex is required")
            color = tuple(int(value) for value in raw_color)
        if any(not 0 <= value <= 255 for value in color):
            raise ManifestError(f"{group_id}: color_rgb values must be within 0..255")
        label = str(item.get("group_label") or item.get("label") or "").strip()
        if not label:
            raise ManifestError(f"{group_id}: caller-supplied label is required")
        groups.append(GroupSpec(group_id, expected_role, label, str(item.get("short_label") or label), color))
    designated = [group for group in groups if group.role == "designated_anchor"]
    comparisons = [group for group in groups if group.role == "comparison"]
    if len(designated) != 1 or designated[0].group_id != anchor_group:
        raise ManifestError("anchor_group must resolve to the single designated_anchor role")
    if [group.group_id for group in comparisons] != reference_groups:
        raise ManifestError("clip manifest comparison groups must match comparison.reference_groups order")

    raw_clips = clip_payload.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ManifestError("clips must be a non-empty array")
    clips: list[ClipSpec] = []
    clip_ids: set[str] = set()
    group_ids = {group.group_id for group in groups}
    for index, item in enumerate(raw_clips):
        if not isinstance(item, dict):
            raise ManifestError(f"clips[{index}] must be an object")
        clip_id = str(item.get("clip_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", clip_id) or clip_id in clip_ids:
            raise ManifestError(f"clips[{index}].clip_id is invalid or duplicate")
        clip_ids.add(clip_id)
        group_id = str(item.get("group_id", "")).strip()
        if group_id not in group_ids:
            raise ManifestError(f"{clip_id}: undeclared group_id")
        start = _finite(item.get("start_s"), f"{clip_id}.start_s")
        end = _finite(item.get("end_s"), f"{clip_id}.end_s")
        if start < 0 or end <= start:
            raise ManifestError(f"{clip_id}: require 0 <= start_s < end_s")
        source = _relative_file(clip_manifest.parent, item.get("source"), f"{clip_id}.source")
        label_origin = str(item.get("label_origin") or "").strip()
        if not label_origin:
            raise ManifestError(f"{clip_id}: label_origin is required")
        clips.append(
            ClipSpec(
                clip_id,
                group_id,
                str(item.get("clip_label") or item.get("label") or clip_id),
                str(item.get("language_label") or item.get("language") or "unspecified"),
                label_origin,
                source,
                start,
                end,
            )
        )
    counts = {group_id: sum(clip.group_id == group_id for clip in clips) for group_id in group_ids}
    if counts[anchor_group] != 1:
        raise ManifestError("the designated anchor group must contain exactly one clip")
    if any(counts[group.group_id] < 1 for group in comparisons):
        raise ManifestError("every comparison group must contain at least one clip")

    timeline = payload.get("timeline", {})
    if not isinstance(timeline, dict):
        raise ManifestError("timeline must be an object")
    intro = _finite(timeline.get("intro_seconds", 5.0), "timeline.intro_seconds")
    gap = _finite(timeline.get("gap_seconds", 0.75), "timeline.gap_seconds")
    outro = _finite(timeline.get("outro_seconds", 5.0), "timeline.outro_seconds")
    fade = _finite(timeline.get("edge_fade_ms", 10.0), "timeline.edge_fade_ms")
    if min(intro, gap, outro, fade) < 0:
        raise ManifestError("timeline values must be non-negative")

    output = payload.get("output")
    if not isinstance(output, dict) or (
        output.get("width"), output.get("height"), output.get("fps"), output.get("audio_rate")
    ) != (WIDTH, HEIGHT, FPS, AUDIO_RATE):
        raise ManifestError("output must declare 1920x1080, 24 fps, and 48000 Hz")

    provenance: list[dict[str, str]] = [
        {"role": "clip_manifest", "path": str(clip_manifest), "sha256": clip_manifest_hash}
    ]
    roles: set[str] = {"clip_manifest"}
    for index, item in enumerate(payload.get("provenance_manifests", [])):
        if not isinstance(item, dict):
            raise ManifestError(f"provenance_manifests[{index}] must be an object")
        role = str(item.get("role", "")).strip()
        if not role or role in roles:
            raise ManifestError("provenance roles must be non-empty and unique")
        roles.add(role)
        resolved = _relative_file(manifest.parent, item.get("path"), f"provenance {role}")
        actual = sha256(resolved)
        if item.get("sha256") != actual:
            raise ManifestError(f"provenance hash mismatch: {role}")
        provenance.append({"role": role, "path": str(resolved), "sha256": actual})

    raw_acoustic = payload.get("feature_artifacts")
    if not isinstance(raw_acoustic, dict):
        raise ManifestError("feature_artifacts must reference validated authoritative artifacts")
    acoustic_artifacts: dict[str, dict[str, str]] = {}
    for key in ("artifact_manifest", "acoustic_features"):
        item = raw_acoustic.get(key)
        path_value = item.get("path") if isinstance(item, dict) else item
        resolved = _relative_file(manifest.parent, path_value, f"feature artifact {key}")
        actual = sha256(resolved)
        if isinstance(item, dict) and item.get("sha256") not in {None, actual}:
            raise ManifestError(f"feature artifact hash mismatch: {key}")
        acoustic_artifacts[key] = {"path": str(resolved), "sha256": actual}

    limitation = str(payload.get("limitation_text") or DEFAULT_LIMITATION).strip()
    if not limitation:
        raise ManifestError("limitation_text must not be empty")
    return RenderSpec(
        title=str(payload.get("title") or "音声信号の特徴比較"),
        limitation_text=limitation,
        source_manifest=manifest,
        groups=tuple(groups),
        clips=tuple(clips),
        intro_s=intro,
        gap_s=gap,
        outro_s=outro,
        edge_fade_ms=fade,
        provenance=tuple(provenance),
        anchor_group=anchor_group,
        acoustic_artifacts=acoustic_artifacts,
        clip_manifest=clip_manifest,
        clip_manifest_sha256=clip_manifest_hash,
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc" if bold else "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def apply_edge_fade(audio: np.ndarray, milliseconds: float) -> np.ndarray:
    result = np.array(audio, dtype=np.float32, copy=True)
    count = min(result.shape[1] // 2, int(round(milliseconds * AUDIO_RATE / 1000.0)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, endpoint=True, dtype=np.float32)
        result[:, :count] *= ramp
        result[:, -count:] *= ramp[::-1]
    return result


def waveform_bins(audio: np.ndarray, count: int) -> np.ndarray:
    mono = np.mean(audio, axis=0)
    edges = np.linspace(0, mono.size, count + 1).astype(int)
    result = np.zeros((count, 2), dtype=np.float32)
    for index in range(count):
        block = mono[edges[index] : edges[index + 1]]
        if block.size:
            result[index] = (float(np.min(block)), float(np.max(block)))
    peak = max(float(np.max(np.abs(result))), 1e-9)
    return result / peak


def draw_limitation_strip(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    top = HEIGHT - 84
    draw.rounded_rectangle((10, top, WIDTH - 10, HEIGHT - 10), radius=11, fill=(12, 25, 34), outline=(88, 105, 117), width=2)
    draw.ellipse((34, top + 16, 74, top + 56), outline=TEXT, width=2)
    draw.text((54, top + 25), "i", anchor="ma", font=font(20, True), fill=TEXT)
    draw.text((94, top + 20), text, font=font(24, True), fill=TEXT)
    draw.text((760, top + 26), LIMITATION_DETAIL, font=font(17), fill=MUTED)


def limitation_reference_strip(text: str) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw_limitation_strip(image, text)
    return np.asarray(image)[HEIGHT - 84 :, :, :]
