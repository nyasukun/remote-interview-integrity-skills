#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw

RENDERING_DIR = Path(__file__).resolve().parent
if str(RENDERING_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERING_DIR))

from adapter import display_metrics
from common import (
    AUDIO_RATE,
    BACKGROUND,
    FPS,
    GRID,
    HEIGHT,
    LIMITATION_DETAIL,
    MUTED,
    PANEL,
    SAMPLES_PER_FRAME,
    TEXT,
    WIDTH,
    ClipSpec,
    GroupSpec,
    RenderSpec,
    apply_edge_fade,
    draw_limitation_strip,
    font,
    load_manifest,
    sha256,
    waveform_bins,
)
from layout_reference import verify_approved_preview, verify_layout_reference
TOOLKIT_DIR = RENDERING_DIR.parent
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from acoustics.model import json_ready  # noqa: E402
from acoustics.signal_features import FeatureConfig, analyze_signal, decode_audio  # noqa: E402
from acoustics.verify_artifacts import verify_acoustic_dir  # noqa: E402


@dataclass
class ClipData:
    spec: ClipSpec
    audio: np.ndarray
    metrics: dict[str, Any]
    decode: dict[str, Any]
    waveform: np.ndarray
    spectrogram: Image.Image
    output_start_sample: int = 0
    output_end_sample: int = 0


@dataclass(frozen=True)
class DeltaGlyphGeometry:
    low_x: int
    high_x: int
    median_x: int
    range_y: int
    median_y: int
    cap_top: int
    cap_bottom: int
    diamond: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PointGlyphGeometry:
    x: int
    y: int
    radius: int


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value: Any = metrics
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _spectrogram(values: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, (8, 99)) if finite.size else (-100.0, 0.0)
    normalized = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    normalized = normalized[::-1]
    base = np.asarray((3, 9, 16), dtype=np.float64)
    tint = np.asarray(color, dtype=np.float64)
    rgb = base + normalized[..., None] ** 0.72 * tint
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _summary(spec: RenderSpec, clips: list[ClipData]) -> dict[str, Any]:
    fields = {
        "pitch_median_hz": "Hz",
        "periodicity_median": "",
        "active_fraction": "",
        "voiced_fraction": "",
        "rms_dbfs": "dBFS",
        "band_ratio.low_0_500": "",
        "band_ratio.mid_500_2000": "",
        "band_ratio.high_2000_8000": "",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "anchor_group": spec.anchor_group,
        "delta_definition": "comparison group metric minus designated anchor metric",
        "policy": "Per-metric signed deltas only; no composite, distance, ranking, winner, identity, nationality, affiliation, intent, or deception inference.",
        "metrics": {},
    }
    for key, unit in fields.items():
        rows: dict[str, Any] = {}
        for group in spec.groups:
            values = [value for clip in clips if clip.spec.group_id == group.group_id if (value := _metric(clip.metrics, key)) is not None]
            rows[group.group_id] = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "median": float(np.median(values)) if values else None,
            }
        point = rows[spec.anchor_group]["median"]
        if point is None:
            raise ValueError(f"designated anchor metric is unavailable: {key}")
        deltas = {
            group_id: {
                statistic: value - point if value is not None and point is not None else None
                for statistic, value in (("min", row["min"]), ("max", row["max"]), ("median", row["median"]))
            }
            for group_id, row in rows.items()
        }
        result["metrics"][key] = {"unit": unit, "groups": rows, "signed_deltas_to_designated": deltas}
    return result


def _box(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], color: tuple[int, int, int] = GRID, width: int = 2) -> None:
    draw.rounded_rectangle(bounds, radius=10, fill=PANEL, outline=color, width=width)


def _waveform(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], values: np.ndarray, color: tuple[int, int, int], progress: float | None) -> None:
    left, top, right, bottom = bounds
    middle = (top + bottom) // 2
    for x in range(right - left):
        index = min(len(values) - 1, int(x / max(right - left - 1, 1) * (len(values) - 1)))
        lo, hi = values[index]
        draw.line((left + x, middle - int(hi * 48), left + x, middle - int(lo * 48)), fill=color)
    if progress is not None:
        x = left + int(np.clip(progress, 0.0, 1.0) * (right - left))
        draw.line((x, top, x, bottom), fill=color, width=3)


def _panel(image: Image.Image, data: ClipData, group: GroupSpec, bounds: tuple[int, int, int, int], active: bool, progress: float | None) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    border = group.color_rgb if active else tuple(int(value * 0.48) for value in group.color_rgb)
    _box(draw, bounds, border, 3 if active else 2)
    draw.text((left + 26, top + 22), group.label, font=font(30, True), fill=group.color_rgb)
    draw.text((left + 28, top + 68), data.spec.label, font=font(18), fill=TEXT)
    draw.text((left + 28, top + 98), f"language  {data.spec.language}", font=font(16), fill=MUTED)
    draw.text((left + 28, top + 126), f"duration  {data.spec.duration_s:.2f}s", font=font(15), fill=MUTED)
    state = "playing" if active else ("fixed anchor" if group.role == "designated_anchor" else "reference")
    draw.text((right - 176, top + 28), state, font=font(17, True), fill=group.color_rgb if active or group.role == "designated_anchor" else MUTED)
    _waveform(draw, (left + 82, top + 150, right - 18, top + 273), data.waveform, group.color_rgb, progress if active else None)
    draw.text((left + 18, top + 164), "waveform", font=font(14, True), fill=MUTED)
    spectrum_box = (left + 82, top + 320, right - 18, bottom - 20)
    image.paste(data.spectrogram.resize((spectrum_box[2] - spectrum_box[0], spectrum_box[3] - spectrum_box[1])), spectrum_box[:2])
    draw.rectangle(spectrum_box, outline=GRID)
    draw.text((left + 16, top + 338), "Log-Mel", font=font(14, True), fill=MUTED)
    if active and progress is not None:
        x = spectrum_box[0] + int(progress * (spectrum_box[2] - spectrum_box[0]))
        draw.line((x, spectrum_box[1], x, spectrum_box[3]), fill=group.color_rgb, width=3)


def _delta_coordinate(value: float, x0: int, x1: int, magnitude: float) -> int:
    return x0 + int(np.clip((value + magnitude) / (2 * magnitude), 0.0, 1.0) * (x1 - x0))


def _draw_range_median_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    low_x: int,
    high_x: int,
    median_x: int,
    axis_y: int,
    color: tuple[int, int, int],
) -> DeltaGlyphGeometry:
    """Draw truthful x positions while separating range and median vertically.

    A range can legitimately collapse below one output pixel. End caps keep that
    collapsed range visible, while the median diamond is placed on a separate
    row and connected with a stem. No minimum x-width is imposed, so the shared
    numeric axis remains authoritative.
    """
    range_y = axis_y + 6
    median_y = axis_y - 7
    cap_top, cap_bottom = range_y - 5, range_y + 5
    diamond = (
        (median_x, median_y - 5),
        (median_x + 5, median_y),
        (median_x, median_y + 5),
        (median_x - 5, median_y),
    )
    draw.line((low_x, range_y, high_x, range_y), fill=color, width=3)
    draw.line((low_x, cap_top, low_x, cap_bottom), fill=color, width=2)
    draw.line((high_x, cap_top, high_x, cap_bottom), fill=color, width=2)
    draw.line((median_x, median_y + 5, median_x, range_y), fill=color, width=2)
    draw.polygon(diamond, fill=color, outline=TEXT)
    return DeltaGlyphGeometry(
        low_x=low_x,
        high_x=high_x,
        median_x=median_x,
        range_y=range_y,
        median_y=median_y,
        cap_top=cap_top,
        cap_bottom=cap_bottom,
        diamond=diamond,
    )


def _draw_point_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    axis_y: int,
    color: tuple[int, int, int],
) -> PointGlyphGeometry:
    """Draw one group/metric observation as a point at its truthful x position."""
    radius = 5
    draw.ellipse(
        (x - radius, axis_y - radius, x + radius, axis_y + radius),
        fill=color,
        outline=TEXT,
        width=1,
    )
    return PointGlyphGeometry(x=x, y=axis_y, radius=radius)


def _draw_delta_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    count: int,
    low_x: int,
    high_x: int,
    median_x: int,
    axis_y: int,
    color: tuple[int, int, int],
) -> PointGlyphGeometry | DeltaGlyphGeometry:
    """Dispatch to the analysis-contract glyph without changing x coordinates."""
    if count < 1:
        raise ValueError("delta glyph count must be positive")
    if count == 1:
        return _draw_point_glyph(draw, x=median_x, axis_y=axis_y, color=color)
    return _draw_range_median_glyph(
        draw,
        low_x=low_x,
        high_x=high_x,
        median_x=median_x,
        axis_y=axis_y,
        color=color,
    )


def _delta_chart(image: Image.Image, bounds: tuple[int, int, int, int], title: str, spec: RenderSpec, summary: dict[str, Any], columns: list[tuple[str, str, float, str]], note: str) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    _box(draw, bounds)
    draw.text((left + 16, top + 11), title, font=font(18, True), fill=TEXT)
    designated = next(group for group in spec.groups if group.group_id == spec.anchor_group)
    comparison = list(spec.comparison_groups)
    draw.text((left + 18, top + 39), "designated = 0", font=font(12, True), fill=designated.color_rgb)
    legend_x = left + 115
    for group in comparison:
        draw.ellipse((legend_x, top + 43, legend_x + 7, top + 50), fill=group.color_rgb)
        draw.text((legend_x + 10, top + 39), group.short_label, font=font(11, True), fill=group.color_rgb)
        legend_x += max(62, len(group.short_label) * 10 + 24)
    glyph_color = comparison[0].color_rgb if comparison else MUTED
    glyph_y = top + 62
    draw.ellipse((left + 18, glyph_y - 4, left + 26, glyph_y + 4), fill=glyph_color, outline=TEXT)
    draw.text((left + 32, top + 53), "point (n=1)", font=font(9), fill=MUTED)
    draw.line((left + 108, glyph_y, left + 134, glyph_y), fill=glyph_color, width=2)
    draw.line((left + 108, glyph_y - 4, left + 108, glyph_y + 4), fill=glyph_color, width=2)
    draw.line((left + 134, glyph_y - 4, left + 134, glyph_y + 4), fill=glyph_color, width=2)
    draw.text((left + 140, top + 53), "min–max (n>1)", font=font(9), fill=MUTED)
    diamond_x = left + 224
    draw.polygon(
        ((diamond_x, glyph_y - 4), (diamond_x + 4, glyph_y), (diamond_x, glyph_y + 4), (diamond_x - 4, glyph_y)),
        fill=glyph_color,
        outline=TEXT,
    )
    draw.text((left + 233, top + 53), "median (n>1)", font=font(9), fill=MUTED)
    width = (right - left - 24) / len(columns)
    for column, (label, key, magnitude, fmt) in enumerate(columns):
        x0 = int(left + 16 + column * width)
        x1 = int(left + 8 + (column + 1) * width)
        draw.text(((x0 + x1) / 2, top + 79), label, anchor="ma", font=font(12, True), fill=TEXT)
        zero = (x0 + x1) // 2
        draw.line((zero, top + 96, zero, top + 190), fill=designated.color_rgb, width=3)

        for row_index, group in enumerate(comparison):
            axis_y = top + 116 + row_index * 28
            draw.line((x0, axis_y, x1, axis_y), fill=(45, 58, 68))
            metric = summary["metrics"][key]
            row = metric["signed_deltas_to_designated"][group.group_id]
            if None in (row["min"], row["max"], row["median"]):
                continue
            count = int(metric["groups"][group.group_id]["count"])
            low = _delta_coordinate(row["min"], x0, x1, magnitude)
            high = _delta_coordinate(row["max"], x0, x1, magnitude)
            median = _delta_coordinate(row["median"], x0, x1, magnitude)
            _draw_delta_glyph(
                draw,
                count=count,
                low_x=low,
                high_x=high,
                median_x=median,
                axis_y=axis_y,
                color=group.color_rgb,
            )
        draw.text((x0, top + 200), format(-magnitude, fmt), font=font(10), fill=MUTED)
        draw.text((zero, top + 200), "0", anchor="ma", font=font(10, True), fill=designated.color_rgb)
        draw.text((x1, top + 200), format(magnitude, fmt), anchor="ra", font=font(10), fill=MUTED)
    draw.text((left + 16, bottom - 23), note, font=font(11), fill=MUTED)


def _clip_frame(spec: RenderSpec, clips: list[ClipData], summary: dict[str, Any], sample: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((WIDTH // 2, 12), spec.title, anchor="ma", font=font(24, True), fill=TEXT)
    groups = {group.group_id: group for group in spec.groups}
    active = next((clip for clip in clips if clip.output_start_sample <= sample < clip.output_end_sample), None)
    anchor = next(clip for clip in clips if clip.spec.group_id == spec.anchor_group)
    comparison = [clip for clip in clips if clip.spec.group_id != spec.anchor_group]
    active_comparison = active if active in comparison else None
    shown = active_comparison or max((clip for clip in comparison if clip.output_end_sample <= sample), key=lambda item: item.output_end_sample, default=comparison[0])

    def progress(clip: ClipData) -> float | None:
        return (sample - clip.output_start_sample) / max(1, clip.output_end_sample - clip.output_start_sample) if clip is active else None

    _panel(image, anchor, groups[spec.anchor_group], (10, 48, 950, 702), anchor is active, progress(anchor))
    _panel(image, shown, groups[shown.spec.group_id], (970, 48, 1910, 702), shown is active, progress(shown))
    _delta_chart(image, (10, 716, 900, 984), "F0 / periodicity", spec, summary, [("ΔF0 Hz", "pitch_median_hz", 120.0, ".0f"), ("Δperiodicity", "periodicity_median", 0.5, ".1f")], "Each delta is one metric minus designated; not a distance or ranking.")
    _delta_chart(image, (918, 716, 1408, 984), "activity / voicing / RMS", spec, summary, [("Δactive", "active_fraction", 0.6, ".1f"), ("Δvoiced", "voiced_fraction", 0.6, ".1f"), ("ΔRMS dB", "rms_dbfs", 12.0, ".0f")], "Shared domains; no metric combination.")
    _delta_chart(image, (1426, 716, 1910, 984), "band fractions", spec, summary, [("Δ0–500", "band_ratio.low_0_500", 0.6, ".1f"), ("Δ0.5–2k", "band_ratio.mid_500_2000", 0.6, ".1f"), ("Δ2–8k", "band_ratio.high_2000_8000", 0.6, ".1f")], "Descriptive only; confounds remain.")
    draw_limitation_strip(image, spec.limitation_text)
    return image


def _card(spec: RenderSpec, outro: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index, group in enumerate(spec.groups):
        draw.rectangle((index * 10, 0, index * 10 + 10, HEIGHT), fill=group.color_rgb)
    draw.text((130, 196), "Comparison complete" if outro else spec.title, font=font(52, True), fill=TEXT)
    draw.text((134, 292), "designated anchor fixed / " + " / ".join(group.label for group in spec.comparison_groups), font=font(25), fill=MUTED)
    draw.line((134, 366, 1780, 366), fill=GRID, width=2)
    draw.text((134, 420), spec.limitation_text, font=font(42, True), fill=TEXT)
    if outro:
        draw.text((134, 545), "Similar and different features do not establish speaker identity.", font=font(28, True), fill=TEXT)
        draw.text((134, 608), "Language, content, emotion, microphone, processing, and codec are confounds.", font=font(24), fill=MUTED)
        draw.text((134, 670), "Offsets are per-metric only—not composite similarity, distance, rank, or a winner.", font=font(23), fill=MUTED)
    else:
        draw.text((134, 545), "waveform / Log-Mel / F0 / periodicity / activity / voicing / RMS / band fractions", font=font(25), fill=MUTED)
        draw.text((134, 608), "Original clip audio plays sequentially without per-clip loudness normalization.", font=font(23), fill=MUTED)
    draw_limitation_strip(image, spec.limitation_text)
    return image


class Encoder:
    def __init__(self, output: Path, limitation: str) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(output), mode="w")
        self.container.metadata["comment"] = f"{limitation} {LIMITATION_DETAIL}"
        self.video = self.container.add_stream("libx264", rate=FPS)
        self.video.width, self.video.height, self.video.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        self.video.options = {"preset": "fast", "crf": "20", "tune": "zerolatency"}
        self.audio = self.container.add_stream("aac", rate=AUDIO_RATE)
        self.audio.layout, self.audio.bit_rate = "stereo", 192_000
        self.frame_index = 0
        self.audio_pts = 0
        self.pending = np.zeros((2, 0), dtype=np.float32)

    def add(self, image: Image.Image, audio: np.ndarray) -> None:
        frame = av.VideoFrame.from_image(image)
        frame.pts, frame.time_base = self.frame_index, Fraction(1, FPS)
        self.frame_index += 1
        for packet in self.video.encode(frame):
            self.container.mux(packet)
        self.pending = np.concatenate((self.pending, audio), axis=1)
        while self.pending.shape[1] >= 1024:
            self._audio(self.pending[:, :1024])
            self.pending = self.pending[:, 1024:]

    def _audio(self, block: np.ndarray) -> None:
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block, dtype=np.float32), format="fltp", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = AUDIO_RATE, self.audio_pts, Fraction(1, AUDIO_RATE)
        self.audio_pts += block.shape[1]
        for packet in self.audio.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.pending.shape[1]:
            self._audio(np.pad(self.pending, ((0, 0), (0, 1024 - self.pending.shape[1]))))
        for packet in self.video.encode():
            self.container.mux(packet)
        for packet in self.audio.encode():
            self.container.mux(packet)
        self.container.close()


def _timeline(spec: RenderSpec, clips: list[ClipData]) -> tuple[np.ndarray, list[dict[str, Any]], float, float]:
    parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    cursor = 0

    def silence(phase: str, seconds: float) -> None:
        nonlocal cursor
        count = int(round(seconds * AUDIO_RATE))
        if not count:
            return
        start = cursor
        parts.append(np.zeros((2, count), dtype=np.float32))
        cursor += count
        rows.append({"phase": phase, "clip_id": "", "group_id": "", "output_start_sample": start, "output_end_sample_exclusive": cursor, "source_start_s": "", "source_end_s": "", "edge_fade_ms": 0.0})

    silence("intro", spec.intro_s)
    for index, clip in enumerate(clips):
        clip.output_start_sample = cursor
        audio = apply_edge_fade(clip.audio, spec.edge_fade_ms)
        parts.append(audio)
        cursor += audio.shape[1]
        clip.output_end_sample = cursor
        rows.append({"phase": "clip", "clip_id": clip.spec.clip_id, "group_id": clip.spec.group_id, "output_start_sample": clip.output_start_sample, "output_end_sample_exclusive": cursor, "source_start_s": clip.spec.start_s, "source_end_s": clip.spec.end_s, "edge_fade_ms": spec.edge_fade_ms})
        if index + 1 < len(clips):
            silence("gap", spec.gap_s)
    silence("outro", spec.outro_s)
    timeline = np.concatenate(parts, axis=1)
    padded_count = math.ceil(timeline.shape[1] / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    if padded_count > timeline.shape[1]:
        start = timeline.shape[1]
        timeline = np.pad(timeline, ((0, 0), (0, padded_count - start)))
        rows.append({"phase": "terminal_pad", "clip_id": "", "group_id": "", "output_start_sample": start, "output_end_sample_exclusive": padded_count, "source_start_s": "", "source_end_s": "", "edge_fade_ms": 0.0})
    peak = float(np.max(np.abs(timeline)))
    gain = 0.98 / peak if peak > 0.999 else 1.0
    timeline *= np.float32(gain)
    for row in rows:
        row["applied_global_gain"] = gain
    return timeline, rows, peak, gain


def render(manifest: Path, output: Path, effective: Path, frame_map: Path, audio_map: Path) -> dict[str, Any]:
    layout_reference = verify_layout_reference()
    spec = load_manifest(manifest)
    layout_review = verify_approved_preview(spec.source_manifest, layout_reference)
    groups = {group.group_id: group for group in spec.groups}
    acoustic_manifest_path = Path(spec.acoustic_artifacts["artifact_manifest"]["path"])
    acoustic_features_path = Path(spec.acoustic_artifacts["acoustic_features"]["path"])
    if acoustic_manifest_path.name != "artifact_manifest.json" or acoustic_features_path.parent != acoustic_manifest_path.parent:
        raise ValueError("authoritative acoustic artifacts must be in one validated artifact directory")
    acoustic_verification = verify_acoustic_dir(acoustic_manifest_path.parent)
    if acoustic_verification.get("status") != "PASS":
        raise ValueError(f"authoritative acoustic artifact verification failed: {acoustic_verification.get('errors')}")
    artifact_manifest_payload = json.loads(acoustic_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest_payload.get("artifacts", {}).get("acoustic_features.json") != spec.acoustic_artifacts["acoustic_features"]["sha256"]:
        raise ValueError("artifact manifest does not bind the authoritative acoustic_features.json")
    acoustic_payload = json.loads(acoustic_features_path.read_text(encoding="utf-8"))
    if acoustic_payload.get("method", {}).get("config", {}).get("sample_rate") != AUDIO_RATE:
        raise ValueError("authoritative acoustic artifacts must use the renderer's 48 kHz analysis rate")
    authoritative_clips = acoustic_payload.get("clips")
    if not isinstance(authoritative_clips, list):
        raise ValueError("authoritative acoustic payload has no clips")
    authoritative_by_id = {str(item.get("clip_id", "")): item for item in authoritative_clips}
    if set(authoritative_by_id) != {clip.clip_id for clip in spec.clips}:
        raise ValueError("authoritative acoustic clip IDs do not exactly match the render manifest")
    authoritative_groups = {str(item.get("group_id", "")): item for item in acoustic_payload.get("group_summaries", [])}
    if set(authoritative_groups) != set(groups):
        raise ValueError("authoritative acoustic group IDs do not exactly match the render manifest")
    for group in spec.groups:
        authoritative_group = authoritative_groups[group.group_id]
        expected_hex = "#" + "".join(f"{value:02X}" for value in group.color_rgb)
        if authoritative_group.get("group_label") != group.label or str(authoritative_group.get("display_color_hex", "")).upper() != expected_hex:
            raise ValueError(f"{group.group_id}: authoritative label/color mismatch")
    clips: list[ClipData] = []
    match_rows: list[dict[str, Any]] = []
    for clip_spec in spec.clips:
        authoritative = authoritative_by_id[clip_spec.clip_id]
        selection = authoritative.get("selection", {})
        source = authoritative.get("source", {})
        if authoritative.get("group_id") != clip_spec.group_id:
            raise ValueError(f"{clip_spec.clip_id}: authoritative group mismatch")
        if abs(float(selection.get("start_s")) - clip_spec.start_s) > 1e-9 or abs(float(selection.get("end_s")) - clip_spec.end_s) > 1e-9:
            raise ValueError(f"{clip_spec.clip_id}: authoritative interval mismatch")
        if Path(str(source.get("path", ""))).expanduser().resolve() != clip_spec.source or source.get("sha256") != sha256(clip_spec.source):
            raise ValueError(f"{clip_spec.clip_id}: authoritative source mismatch")
        decoded = decode_audio(clip_spec.source, sample_rate=AUDIO_RATE, start_s=clip_spec.start_s, end_s=clip_spec.end_s)
        bundle = analyze_signal(decoded.samples, sample_rate=AUDIO_RATE, config=FeatureConfig(sample_rate=AUDIO_RATE))
        actual_summary = json_ready(bundle.summary)
        expected_summary = authoritative.get("summary")
        if actual_summary != expected_summary:
            raise ValueError(f"{clip_spec.clip_id}: renderer summary differs from authoritative acoustic artifact")
        metrics = {**display_metrics(actual_summary), "authoritative_summary": actual_summary}
        render_samples = decoded.samples
        if render_samples.shape[1] == 1:
            render_samples = np.repeat(render_samples, 2, axis=1)
        expected = int(round(clip_spec.duration_s * AUDIO_RATE))
        if abs(len(render_samples) - expected) > int(0.03 * AUDIO_RATE):
            raise ValueError(f"{clip_spec.clip_id}: decoded coverage differs from requested interval")
        render_samples = np.pad(render_samples, ((0, max(0, expected - len(render_samples))), (0, 0)))[:expected]
        audio = np.ascontiguousarray(render_samples.T, dtype=np.float32)
        clips.append(ClipData(clip_spec, audio, metrics, {"sample_rate": AUDIO_RATE, "source_rate": decoded.source_sample_rate, "source_channels": decoded.source_channels, "source_codec": decoded.source_codec, "decoded_frames": decoded.decoded_frames, "rendered_samples": audio.shape[1]}, waveform_bins(audio, 820), _spectrogram(bundle.log_mel_power_db.T, groups[clip_spec.group_id].color_rgb)))
        match_rows.append({"clip_id": clip_spec.clip_id, "group_id": clip_spec.group_id, "interval_exact_match": True, "source_exact_match": True, "summary_exact_match": True, "canonical_summary_sha256": _canonical_sha256(actual_summary)})
    summary = _summary(spec, clips)
    timeline, audio_rows, source_peak, gain = _timeline(spec, clips)
    output, effective, frame_map, audio_map = (path.expanduser().resolve() for path in (output, effective, frame_map, audio_map))
    encoder = Encoder(output, spec.limitation_text)
    frame_rows: list[dict[str, Any]] = []
    try:
        for frame_index in range(timeline.shape[1] // SAMPLES_PER_FRAME):
            sample = frame_index * SAMPLES_PER_FRAME
            active = next((clip for clip in clips if clip.output_start_sample <= sample < clip.output_end_sample), None)
            if sample < clips[0].output_start_sample:
                image = _card(spec, False)
            elif sample >= clips[-1].output_end_sample:
                image = _card(spec, True)
            else:
                image = _clip_frame(spec, clips, summary, sample)
            encoder.add(image, timeline[:, sample : sample + SAMPLES_PER_FRAME])
            phase_row = next(row for row in audio_rows if int(row["output_start_sample"]) <= sample < int(row["output_end_sample_exclusive"]))
            frame_rows.append({"output_frame_index": frame_index, "output_pts_s": frame_index / FPS, "phase": phase_row["phase"], "clip_id": active.spec.clip_id if active else "", "group_id": active.spec.group_id if active else "", "source_time_s": active.spec.start_s + (sample - active.output_start_sample) / AUDIO_RATE if active else ""})
    finally:
        encoder.close()
    frame_map.parent.mkdir(parents=True, exist_ok=True)
    with frame_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_rows[0].keys())
        writer.writeheader(); writer.writerows(frame_rows)
    audio_map.parent.mkdir(parents=True, exist_ok=True)
    with audio_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audio_rows[0].keys())
        writer.writeheader(); writer.writerows(audio_rows)
    clip_payload = []
    for clip in clips:
        clip_payload.append({"clip_id": clip.spec.clip_id, "group_id": clip.spec.group_id, "label": clip.spec.label, "language": clip.spec.language, "label_origin": clip.spec.label_origin, "source": str(clip.spec.source), "source_sha256": sha256(clip.spec.source), "source_start_s": clip.spec.start_s, "source_end_s": clip.spec.end_s, "output_start_sample": clip.output_start_sample, "output_end_sample_exclusive": clip.output_end_sample, "decode": clip.decode, "metrics": clip.metrics})
    payload = {
        "schema_version": 1,
        "renderer": "rendering/render.py",
        "source_manifest": str(spec.source_manifest),
        "source_manifest_sha256": sha256(spec.source_manifest),
        "clip_manifest": {"path": str(spec.clip_manifest), "sha256": spec.clip_manifest_sha256},
        "provenance_manifests": list(spec.provenance),
        "layout": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "pixel_format": "yuv420p", "mode": "designated_point_centered", "reference_asset": layout_reference, "approved_preview": layout_review},
        "comparison": {"anchor_group": spec.anchor_group, "display_anchor_group": spec.anchor_group, "point_group": spec.anchor_group, "reference_groups": [group.group_id for group in spec.comparison_groups], "delta_definition": "group metric minus user-designated point metric", "shared_metric_scales": True},
        "groups": [{"group_id": group.group_id, "role": group.role, "label": group.label, "short_label": group.short_label, "color_rgb": list(group.color_rgb), "clip_count": sum(clip.spec.group_id == group.group_id for clip in clips)} for group in spec.groups],
        "audio": {"sample_rate": AUDIO_RATE, "channels": 2, "edge_fade_ms": spec.edge_fade_ms, "source_timeline_peak": source_peak, "applied_global_gain": gain, "sample_count": timeline.shape[1], "policy": "sequential source audio; no per-clip loudness normalization"},
        "limitation": {"permanent_text": spec.limitation_text, "applies_to_every_frame": True, "detail": LIMITATION_DETAIL},
        "feature_policy": "descriptive time/frequency/autocorrelation metrics only; no speaker embedding, identity model, similarity, composite, rank, or winner",
        "authoritative_acoustic_match": {"artifacts": spec.acoustic_artifacts, "artifact_directory_verification": acoustic_verification, "all_intervals_exact_match": True, "all_sources_exact_match": True, "all_summaries_exact_match": True, "clips": match_rows},
        "summary": summary,
        "clips": clip_payload,
        "timeline": audio_rows,
        "output": {"video": str(output), "sha256": sha256(output), "frame_map": str(frame_map), "frame_map_sha256": sha256(frame_map), "audio_map": str(audio_map), "audio_map_sha256": sha256(audio_map), "frame_count": len(frame_rows), "duration_s": len(frame_rows) / FPS},
    }
    effective.parent.mkdir(parents=True, exist_ok=True)
    effective.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a designated-anchor descriptive voice-signal comparison.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-manifest", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, required=True)
    parser.add_argument("--audio-map", type=Path, required=True)
    args = parser.parse_args()
    render(args.manifest, args.output, args.effective_manifest, args.frame_map, args.audio_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
