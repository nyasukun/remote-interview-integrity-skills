#!/usr/bin/env python3
"""Render an auditable bilabial-plosive lip-closure evidence video.

The renderer is deliberately local-only and frame deterministic:

* the normal pass maps one output frame to one native source frame;
* the 0.25x pass repeats each selected native source frame exactly four times;
* the audio is linearly stretched to the same duration (pitch is lowered);
* every output frame is recorded in ``closure_evidence_frame_map.csv``.

The accepted manifest shape is intentionally permissive.  The top-level event
list may be named ``events``, ``cases``, or ``clips``.  See ``load_manifest``
and ``parse_event`` for the supported aliases.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from video_integrity_analyzer.media import probe_video_frame_timing  # noqa: E402
from video_integrity_analyzer.layout_reference import (  # noqa: E402
    load_approved_layout_reference,
    verify_approved_preview,
)


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 24
AUDIO_RATE = 48_000
AUDIO_SAMPLES_PER_VIDEO_FRAME = AUDIO_RATE // OUTPUT_FPS

HEADER_HEIGHT = 72
VIDEO_HEIGHT = 648
WAVEFORM_Y = HEADER_HEIGHT + VIDEO_HEIGHT
WAVEFORM_HEIGHT = OUTPUT_HEIGHT - WAVEFORM_Y

MISSING_COLOR = "#ff4d5a"
REFERENCE_COLOR = "#42d47e"
CYAN = "#58d6ff"
WAVEFORM_COLOR = "#55dbe9"
CLOSURE_WINDOW_COLOR = "#f4dd62"
WHITE = "#f7f9fc"
MUTED = "#a8b3c2"
PANEL = "#121923"
PANEL_2 = "#0c1118"
GRID = "#2b3543"

DEFAULT_MOUTH_ROI = (0.38, 0.50, 0.62, 0.78)
MOUTH_INSET_XYXY = (1396, HEADER_HEIGHT + 26, 1882, HEADER_HEIGHT + 342)
DEFAULT_NOTE = "破裂音前後の口唇運動を比較"
MISSING_LEGEND = "赤：閉鎖確認できず"
REFERENCE_LEGEND = "緑：同期/閉鎖あり参照"
LATE_CONTACT_LIMITATION = "後続音素の口形である可能性あり。"
NORMALIZATION_LIMITATION = "ケース別正規化：絶対音圧比較不可"
CAUSAL_LIMITATION = (
    "本資料は、録画内の音響開放時刻と可視的な口唇接触を並べた観測資料です。"
    "A/V不整合の原因、発話者の本人性、国籍、所属、意図を判定するものではありません。"
)


@dataclass(frozen=True)
class SourceFrame:
    time_s: float
    source_frame_index: int
    image: Image.Image


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    release_s: float
    category: str
    label: str
    note: str
    pre_s: float
    post_s: float
    mouth_roi: tuple[float, float, float, float]
    order: float
    warning: str = ""
    reference_kind: str = "contact"

    @property
    def color(self) -> str:
        return MISSING_COLOR if self.category == "missing" else REFERENCE_COLOR

    @property
    def category_ja(self) -> str:
        if self.category == "missing":
            return "可視フレーム内で明瞭な閉鎖を確認できず"
        if self.reference_kind == "sync":
            return "同期参照（閉鎖あり）"
        return "閉鎖あり参照"

    @property
    def classification(self) -> str:
        if self.category == "missing":
            return "closure_absent"
        return "sync_reference" if self.reference_kind == "sync" else "contact_reference"

    @property
    def start_s(self) -> float:
        return self.release_s - self.pre_s

    @property
    def end_s(self) -> float:
        return self.release_s + self.post_s


@dataclass(frozen=True)
class Manifest:
    title: str
    subtitle: str
    source_video: Path | None
    events: tuple[EvidenceEvent, ...]
    phone_label: str = "/p/"


@dataclass(frozen=True)
class FontSet:
    title: ImageFont.FreeTypeFont | ImageFont.ImageFont
    header: ImageFont.FreeTypeFont | ImageFont.ImageFont
    finding: ImageFont.FreeTypeFont | ImageFont.ImageFont
    body: ImageFont.FreeTypeFont | ImageFont.ImageFont
    small: ImageFont.FreeTypeFont | ImageFont.ImageFont
    tiny: ImageFont.FreeTypeFont | ImageFont.ImageFont
    huge: ImageFont.FreeTypeFont | ImageFont.ImageFont


def _first(mapping: dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and value != "":
            return value
    return default


def _float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _normalize_category(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    missing_values = {
        "missing",
        "absence",
        "absent",
        "no_contact",
        "no_visible_contact",
        "closure_missing",
        "closure_absence",
        "閉鎖欠如",
        "閉鎖なし",
        "欠如",
    }
    reference_values = {
        "reference",
        "contact_reference",
        "contact",
        "contact_present",
        "strong_contact",
        "閉鎖あり参照",
        "参照",
    }
    sync_values = {"sync_reference", "sync", "synchronized", "同期参照", "同期"}
    if text in missing_values or any(token in text for token in ("missing", "absence", "absent", "欠如")):
        return "missing"
    if text in reference_values or text in sync_values or any(
        token in text for token in ("reference", "sync", "参照", "同期")
    ):
        return "reference"
    raise ValueError(f"unrecognized event category: {value!r}")


def _normalize_roi(value: Any) -> tuple[float, float, float, float]:
    if value is None:
        return DEFAULT_MOUTH_ROI
    if isinstance(value, dict):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            raw = (value["x0"], value["y0"], value["x1"], value["y1"])
        elif all(key in value for key in ("x", "y", "w", "h")):
            raw = (
                value["x"],
                value["y"],
                _float(value["x"], name="mouth_roi.x") + _float(value["w"], name="mouth_roi.w"),
                _float(value["y"], name="mouth_roi.y") + _float(value["h"], name="mouth_roi.h"),
            )
        else:
            raise ValueError("mouth_roi object needs x0/y0/x1/y1 or x/y/w/h")
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        raw = value
    else:
        raise ValueError(f"mouth_roi must contain four coordinates, got {value!r}")
    roi = tuple(_float(component, name="mouth_roi") for component in raw)
    # Pixel ROIs are accepted relative to the deterministic 1920x1080 canvas.
    if max(roi) > 1.5:
        roi = (
            roi[0] / OUTPUT_WIDTH,
            roi[1] / OUTPUT_HEIGHT,
            roi[2] / OUTPUT_WIDTH,
            roi[3] / OUTPUT_HEIGHT,
        )
    x0, y0, x1, y1 = roi
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"mouth_roi must be inside the source frame, got {roi!r}")
    return (x0, y0, x1, y1)


def parse_event(raw: dict[str, Any], index: int, defaults: dict[str, Any]) -> EvidenceEvent:
    release = _first(
        raw,
        (
            "source_release_s",
            "release_s",
            "release_time_s",
            "audio_release_time_s",
            "burst_time_s",
            "marker_time_s",
            "time_s",
        ),
    )
    if release is None:
        raise ValueError(f"event {index} has no release/burst time")
    window = raw.get("window") if isinstance(raw.get("window"), dict) else {}
    clip = raw.get("clip") if isinstance(raw.get("clip"), dict) else {}
    start = _first(raw, ("start_s", "clip_start_s"), _first(window, ("start_s", "start"), _first(clip, ("start_s", "start"))))
    end = _first(raw, ("end_s", "clip_end_s"), _first(window, ("end_s", "end"), _first(clip, ("end_s", "end"))))
    release_s = _float(release, name=f"event {index} release")
    if start is not None and end is not None:
        pre_s = release_s - _float(start, name=f"event {index} start")
        post_s = _float(end, name=f"event {index} end") - release_s
    else:
        pre_s = _float(
            _first(raw, ("pre_s", "before_s", "window_before_s"), _first(window, ("pre_s", "before_s"), defaults["pre_s"])),
            name=f"event {index} pre_s",
        )
        post_s = _float(
            _first(raw, ("post_s", "after_s", "window_after_s"), _first(window, ("post_s", "after_s"), defaults["post_s"])),
            name=f"event {index} post_s",
        )
    if pre_s <= 0.0 or post_s <= 0.0:
        raise ValueError(f"event {index} window must straddle release time")
    event_id = str(_first(raw, ("event_id", "runner_event_id", "id", "case_id"), f"case-{index:03d}"))
    category_raw = _first(raw, ("category", "kind", "type", "classification", "status", "closure_category"))
    category = _normalize_category(category_raw)
    normalized_category_raw = str(category_raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    sync_values = {"sync_reference", "sync", "synchronized", "同期参照", "同期"}
    reference_kind = "sync" if category == "reference" and normalized_category_raw in sync_values else "contact"
    label = str(
        _first(
            raw,
            (
                "label",
                "title",
                "audio_context",
                "target_word_or_phrase",
                "target_token",
                "context",
                "token_text",
                "token",
            ),
            event_id,
        )
    )
    visual_evidence = raw.get("visual_evidence") if isinstance(raw.get("visual_evidence"), dict) else {}
    note = str(
        _first(
            raw,
            ("note", "annotation", "description", "finding"),
            _first(
                visual_evidence,
                ("summary",),
                _first(raw, ("selection_rationale",), DEFAULT_NOTE),
            ),
        )
    )
    late_contact_offset = _first(
        visual_evidence,
        (
            "first_clear_contact_offset_ms_through_plus500",
            "first_clear_contact_offset_ms",
        ),
    )
    if category == "missing" and late_contact_offset is not None:
        late_ms = _float(late_contact_offset, name=f"event {index} late contact offset")
        if late_ms > 0.0:
            note = (
                f"音響開放後+{late_ms:.1f} msに接触候補。"
                f"{LATE_CONTACT_LIMITATION}"
            )
    roi_value = _first(raw, ("mouth_roi", "mouth_bbox", "roi"), defaults.get("mouth_roi"))
    order = _float(_first(raw, ("order", "case_order", "index"), index), name=f"event {index} order")
    warning = str(
        _first(
            raw,
            ("warning", "caution", "presentation_warning", "display_warning"),
            "",
        )
    ).strip()
    return EvidenceEvent(
        event_id=event_id,
        release_s=release_s,
        category=category,
        label=label,
        note=note,
        pre_s=pre_s,
        post_s=post_s,
        mouth_roi=_normalize_roi(roi_value),
        order=order,
        warning=warning,
        reference_kind=reference_kind,
    )


def load_manifest(path: Path) -> Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        root: dict[str, Any] = {}
        raw_events = payload
    elif isinstance(payload, dict):
        root = payload
        raw_events = _first(root, ("events", "cases", "clips"))
        if raw_events is None and isinstance(root.get("manifest"), dict):
            nested = root["manifest"]
            raw_events = _first(nested, ("events", "cases", "clips"))
            root = {**root, **nested}
    else:
        raise ValueError("manifest root must be an object or event list")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("manifest must contain a non-empty events/cases/clips list")
    defaults_raw = root.get("defaults") if isinstance(root.get("defaults"), dict) else {}
    defaults = {
        "pre_s": _first(defaults_raw, ("pre_s", "before_s"), _first(root, ("pre_s", "window_before_s"), 0.8)),
        "post_s": _first(defaults_raw, ("post_s", "after_s"), _first(root, ("post_s", "window_after_s"), 0.8)),
        "mouth_roi": _first(defaults_raw, ("mouth_roi", "mouth_bbox", "roi"), _first(root, ("mouth_roi", "mouth_bbox", "roi"))),
    }
    events = [parse_event(raw, index, defaults) for index, raw in enumerate(raw_events, start=1) if isinstance(raw, dict)]
    if len(events) != len(raw_events):
        raise ValueError("every manifest event must be an object")
    event_ids = [event.event_id for event in events]
    if any(not event_id.strip() for event_id in event_ids):
        raise ValueError("every manifest event must have a non-empty event ID")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(f"manifest event IDs must be unique: {event_ids}")
    # Presentation order is independent from the red/green classification.
    # Source chronology prevents the edit itself from grouping one result
    # class before the other and is deterministic across manifest revisions.
    events.sort(key=lambda event: (event.release_s, event.order, event.event_id))
    source_value = _first(root, ("source_video", "video", "source"))
    source_video = None
    if source_value:
        source_video = Path(str(source_value)).expanduser()
        if not source_video.is_absolute():
            source_video = (path.parent / source_video).resolve()
    return Manifest(
        title=str(_first(root, ("title", "name"), "両唇破裂音と口唇閉鎖の視覚比較")),
        subtitle=str(_first(root, ("subtitle", "description"), "破裂音と口唇閉鎖の時間対応")),
        source_video=source_video,
        events=tuple(events),
        phone_label=str(_first(root, ("phone_label", "phoneme_label", "phone"), "/p/")),
    )


def _font_path() -> Path | None:
    candidates = [
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"),
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _font(size: int, path: Path | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_fonts() -> FontSet:
    path = _font_path()
    return FontSet(
        title=_font(38, path),
        header=_font(30, path),
        finding=_font(23, path),
        body=_font(25, path),
        small=_font(20, path),
        tiny=_font(16, path),
        huge=_font(62, path),
    )


def _fit_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    target_ratio = width / height
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        crop_width = int(round(image.height * target_ratio))
        left = (image.width - crop_width) // 2
        crop = image.crop((left, 0, left + crop_width, image.height))
    else:
        crop_height = int(round(image.width / target_ratio))
        top = (image.height - crop_height) // 2
        crop = image.crop((0, top, image.width, top + crop_height))
    return crop.resize((width, height), Image.Resampling.BILINEAR)


def _rgba_overlay(base: Image.Image, rectangle: tuple[int, int, int, int], color: str, alpha: int) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(rectangle, radius=14, fill=(*_hex_rgb(color), alpha))
    base.alpha_composite(layer)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index:index + 2], 16) for index in (0, 2, 4))


def _source_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def _ellipsize_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "…"
    shortened = text
    while shortened and draw.textbbox((0, 0), shortened + suffix, font=font)[2] > max_width:
        shortened = shortened[:-1]
    return shortened + suffix


def decode_video_window(path: Path, event: EvidenceEvent) -> tuple[list[SourceFrame], float]:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate_value = stream.average_rate or stream.base_rate or stream.guessed_rate
        rate = float(rate_value) if rate_value else float(OUTPUT_FPS)
        if abs(rate - OUTPUT_FPS) > 0.01:
            raise RuntimeError(f"source must be 24 fps for frame-auditable slow motion; got {rate:.6f}")
        time_base = float(stream.time_base)
        start_pts = int(stream.start_time or 0)
        seek_s = max(0.0, event.start_s - 1.0)
        container.seek(int(seek_s / time_base), stream=stream, backward=True, any_frame=False)
        margin = 1.5 / rate
        decoded: list[SourceFrame] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time_s = float(frame.pts * frame.time_base)
            if time_s < event.start_s - margin:
                continue
            if time_s > event.end_s + margin:
                break
            absolute_index = int(round((int(frame.pts) - start_pts) * time_base * rate))
            decoded.append(SourceFrame(time_s, absolute_index, frame.to_image().convert("RGB")))
    if not decoded:
        raise RuntimeError(f"no source frames decoded for {event.event_id}")
    return decoded, rate


def select_native_frames(frames: Sequence[SourceFrame], event: EvidenceEvent) -> list[SourceFrame]:
    frame_count = int(round((event.pre_s + event.post_s) * OUTPUT_FPS))
    if frame_count <= 0:
        raise ValueError(f"empty event window for {event.event_id}")
    times = [frame.time_s for frame in frames]
    selected: list[SourceFrame] = []
    for index in range(frame_count):
        target = event.start_s + index / OUTPUT_FPS
        insertion = bisect.bisect_left(times, target)
        choices = [candidate for candidate in (insertion - 1, insertion) if 0 <= candidate < len(frames)]
        nearest = min(choices, key=lambda candidate: abs(times[candidate] - target))
        selected.append(frames[nearest])
    # A 24 fps CFR source should map every normal output frame to a distinct
    # native frame.  Failing here prevents an unnoticed timing substitution.
    indices = [frame.source_frame_index for frame in selected]
    if len(set(indices)) != len(indices):
        raise RuntimeError(f"normal pass for {event.event_id} did not map 1:1 to native frames: {indices}")
    if any(right - left != 1 for left, right in zip(indices, indices[1:])):
        raise RuntimeError(f"normal pass for {event.event_id} skipped a native frame: {indices}")
    return selected


def _audio_array(frame: av.AudioFrame) -> np.ndarray:
    values = frame.to_ndarray()
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[0] not in (1, 2) and values.shape[1] in (1, 2):
        values = values.T
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        values = values.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        values = values.astype(np.float32, copy=False)
    if values.shape[0] == 1:
        values = np.repeat(values, 2, axis=0)
    return np.ascontiguousarray(values[:2])


def decode_audio_window(path: Path, event: EvidenceEvent) -> np.ndarray:
    sample_count = int(round((event.pre_s + event.post_s) * AUDIO_RATE))
    result = np.zeros((2, sample_count), dtype=np.float32)
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        source_rate = int(stream.codec_context.sample_rate or 0)
        if source_rate != AUDIO_RATE:
            raise RuntimeError(f"source audio must be 48 kHz; got {source_rate}")
        container.seek(
            int(max(0.0, event.start_s - 1.0) / float(stream.time_base)),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time_s = float(frame.pts * frame.time_base)
            duration_s = frame.samples / float(frame.sample_rate)
            if time_s >= event.end_s:
                break
            if time_s + duration_s <= event.start_s:
                continue
            values = _audio_array(frame)
            destination_start = int(round((time_s - event.start_s) * AUDIO_RATE))
            source_start = max(0, -destination_start)
            destination_start = max(0, destination_start)
            copy_count = min(
                values.shape[1] - source_start,
                result.shape[1] - destination_start,
            )
            if copy_count > 0:
                result[:, destination_start:destination_start + copy_count] = values[
                    :, source_start:source_start + copy_count
                ]
    return result


def stretch_audio(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    if audio.shape[1] == target_samples:
        result = audio.astype(np.float32, copy=True)
    elif audio.shape[1] <= 1:
        result = np.zeros((2, target_samples), dtype=np.float32)
    else:
        source_positions = np.linspace(0.0, audio.shape[1] - 1.0, num=target_samples, endpoint=True)
        left = np.floor(source_positions).astype(np.int64)
        right = np.minimum(left + 1, audio.shape[1] - 1)
        fraction = (source_positions - left).astype(np.float32)
        result = audio[:, left] * (1.0 - fraction) + audio[:, right] * fraction
        result = result.astype(np.float32, copy=False)
    fade_samples = min(int(round(0.006 * AUDIO_RATE)), target_samples // 2)
    if fade_samples > 1:
        fade = np.sin(np.linspace(0.0, math.pi / 2.0, fade_samples, dtype=np.float32)) ** 2
        result[:, :fade_samples] *= fade
        result[:, -fade_samples:] *= fade[::-1]
    return np.ascontiguousarray(result)


def waveform_envelope(audio: np.ndarray, width: int) -> np.ndarray:
    mono = np.mean(audio, axis=0)
    if mono.size == 0:
        return np.zeros(width, dtype=np.float32)
    edges = np.linspace(0, mono.size, width + 1).astype(np.int64)
    envelope = np.zeros(width, dtype=np.float32)
    for index in range(width):
        segment = mono[edges[index]:edges[index + 1]]
        if segment.size:
            envelope[index] = float(np.sqrt(np.mean(segment * segment)))
    scale = float(np.percentile(envelope, 99.0)) if np.any(envelope) else 1.0
    if scale <= 1e-9:
        return envelope
    return np.clip(envelope / scale, 0.0, 1.0)


def quantize_event_window(event: EvidenceEvent) -> EvidenceEvent:
    """Quantize each side of the window to whole native 24 fps frames.

    A requested 0.8 seconds is 19.2 frames and therefore cannot coexist with
    a 24 fps CFR output and a strict 1:1 normal-frame mapping.  Nineteen frames
    (0.791666667 s) is the nearest non-overshooting representation.  Recording
    this effective value avoids silently resampling the normal-speed audio.
    """
    pre_frames = max(1, int(math.floor(event.pre_s * OUTPUT_FPS + 0.5)))
    post_frames = max(1, int(math.floor(event.post_s * OUTPUT_FPS + 0.5)))
    return replace(
        event,
        pre_s=pre_frames / OUTPUT_FPS,
        post_s=post_frames / OUTPUT_FPS,
    )


def _draw_waveform(
    canvas: Image.Image,
    event: EvidenceEvent,
    envelope: np.ndarray,
    source_time_s: float,
    phase: str,
    fonts: FontSet,
    phone_label: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, WAVEFORM_Y, OUTPUT_WIDTH, OUTPUT_HEIGHT), fill=PANEL)
    left, right = 72, OUTPUT_WIDTH - 72
    top, bottom = WAVEFORM_Y + 74, OUTPUT_HEIGHT - 100
    center_y = (top + bottom) // 2
    wave_height = (bottom - top) * 0.42
    draw.text((72, WAVEFORM_Y + 20), "音圧波形（正規化）", fill=WHITE, font=fonts.small)
    draw.text(
        (292, WAVEFORM_Y + 22),
        NORMALIZATION_LIMITATION,
        fill=MUTED,
        font=fonts.tiny,
    )
    draw.text((right - 290, WAVEFORM_Y + 21), "固定線＝音響開放（破裂点）", fill=MUTED, font=fonts.tiny)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(left + (right - left) * fraction))
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        offset = -event.pre_s + (event.pre_s + event.post_s) * fraction
        draw.text((x - 25, bottom + 13), f"{offset:+.2f}s", fill=MUTED, font=fonts.tiny)
    draw.line((left, center_y, right, center_y), fill="#526070", width=1)
    xs = np.linspace(left, right, len(envelope), endpoint=False)
    upper = [(int(x), int(round(center_y - amplitude * wave_height))) for x, amplitude in zip(xs, envelope)]
    lower = [(int(x), int(round(center_y + amplitude * wave_height))) for x, amplitude in reversed(list(zip(xs, envelope)))]
    if upper:
        draw.polygon(upper + lower, fill=WAVEFORM_COLOR)
    burst_fraction = event.pre_s / (event.pre_s + event.post_s)
    burst_x = int(round(left + (right - left) * burst_fraction))

    # Keep the pre-release visual review interval distinct from the red/green
    # acoustic release marker.  The bracket is a presentation aid only; the
    # complete audited interval remains stated verbatim below the waveform.
    review_start_offset_s = max(-event.pre_s, -0.250)
    review_start_fraction = (review_start_offset_s + event.pre_s) / (event.pre_s + event.post_s)
    review_start_x = int(round(left + (right - left) * review_start_fraction))
    bracket_y = top - 8
    if review_start_x < burst_x:
        draw.line(
            (review_start_x, bracket_y, burst_x, bracket_y),
            fill=CLOSURE_WINDOW_COLOR,
            width=4,
        )
        draw.line(
            (review_start_x, bracket_y, review_start_x, top + 3),
            fill=CLOSURE_WINDOW_COLOR,
            width=3,
        )
        draw.line(
            (burst_x, bracket_y, burst_x, top + 3),
            fill=CLOSURE_WINDOW_COLOR,
            width=3,
        )
        closure_label = "閉鎖確認窓"
        closure_bbox = draw.textbbox((0, 0), closure_label, font=fonts.small)
        closure_width = closure_bbox[2] - closure_bbox[0]
        draw.text(
            ((review_start_x + burst_x - closure_width) // 2, bracket_y - 31),
            closure_label,
            fill=CLOSURE_WINDOW_COLOR,
            font=fonts.small,
        )

    burst_half_width = max(5, int(round((right - left) * 0.018 / (event.pre_s + event.post_s))))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (burst_x - burst_half_width, top, burst_x + burst_half_width, bottom),
        fill=(*_hex_rgb(event.color), 65),
    )
    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    draw.line((burst_x, WAVEFORM_Y + 57, burst_x, bottom + 2), fill=event.color, width=5)
    draw.polygon(
        [(burst_x, WAVEFORM_Y + 55), (burst_x - 11, WAVEFORM_Y + 38), (burst_x + 11, WAVEFORM_Y + 38)],
        fill=event.color,
    )
    marker_text = f"{phone_label} 音響開放（破裂点）"
    bbox = draw.textbbox((0, 0), marker_text, font=fonts.small)
    draw.text((burst_x - (bbox[2] - bbox[0]) // 2, WAVEFORM_Y + 8), marker_text, fill=event.color, font=fonts.small)
    progress = (source_time_s - event.start_s) / (event.pre_s + event.post_s)
    playhead_x = int(round(left + np.clip(progress, 0.0, 1.0) * (right - left)))
    draw.line((playhead_x, top - 9, playhead_x, bottom + 3), fill=CYAN, width=3)
    draw.ellipse((playhead_x - 6, top - 15, playhead_x + 6, top - 3), fill=CYAN)
    speed_text = "1.0× 通常速度" if phase == "normal" else "0.25× スロー（原フレーム4回保持）"
    draw.rounded_rectangle((72, bottom + 39, 425, bottom + 82), radius=12, fill="#1d2936")
    draw.text((90, bottom + 48), speed_text, fill=CYAN, font=fonts.tiny)
    if event.category == "missing":
        draw.text(
            (458, bottom + 49),
            "監査判定窓：音響開放の250 ms前〜167 ms後（原録画24 fps）",
            fill=MUTED,
            font=fonts.tiny,
        )


def _top_vertical_crop(source: Image.Image) -> tuple[Image.Image, int, float]:
    # The 1920x1080 source is displayed as a centered 1920x648 native-pixel
    # strip.  For any other 16:9 input, use the same normalized crop.
    crop_height = int(round(source.height * VIDEO_HEIGHT / OUTPUT_HEIGHT))
    crop_height = min(source.height, max(1, crop_height))
    top = (source.height - crop_height) // 2
    crop = source.crop((0, top, source.width, top + crop_height))
    if crop.size != (OUTPUT_WIDTH, VIDEO_HEIGHT):
        scale = OUTPUT_WIDTH / crop.width
        crop = crop.resize((OUTPUT_WIDTH, VIDEO_HEIGHT), Image.Resampling.BILINEAR)
    else:
        scale = 1.0
    return crop, top, scale


def _mouth_inset_connectors(
    mapped_roi: tuple[int, int, int, int],
    inset_box: tuple[int, int, int, int],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return two source-ROI-to-inset guide lines clipped to the video strip."""
    roi_right = min(OUTPUT_WIDTH - 1, max(0, mapped_roi[2]))
    roi_top = min(HEADER_HEIGHT + VIDEO_HEIGHT - 1, max(HEADER_HEIGHT, mapped_roi[1]))
    roi_bottom = min(HEADER_HEIGHT + VIDEO_HEIGHT - 1, max(HEADER_HEIGHT, mapped_roi[3]))
    inset_left = inset_box[0]
    inset_top = inset_box[1] + 3
    inset_bottom = inset_box[3] - 3
    return (
        ((roi_right, roi_top), (inset_left, inset_top)),
        ((roi_right, roi_bottom), (inset_left, inset_bottom)),
    )


def render_evidence_frame(
    source_frame: SourceFrame,
    event: EvidenceEvent,
    envelope: np.ndarray,
    phase: str,
    case_order: int,
    case_count: int,
    category_order: int,
    category_count: int,
    fonts: FontSet,
    phone_label: str = "/p/",
) -> Image.Image:
    canvas = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), PANEL_2)
    top_crop, crop_top, crop_scale = _top_vertical_crop(source_frame.image)
    canvas.paste(top_crop, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, OUTPUT_WIDTH, HEADER_HEIGHT), fill="#080c12")
    draw.line((0, HEADER_HEIGHT - 1, OUTPUT_WIDTH, HEADER_HEIGHT - 1), fill=GRID, width=1)

    case_chip = (16, 9, 250, 63)
    draw.rounded_rectangle(case_chip, radius=3, fill="#111821", outline="#566170", width=2)
    draw.text((34, 20), f"ケース {case_order} / {case_count}", fill=WHITE, font=fonts.body)

    title_text = _ellipsize_text(
        draw,
        f"{phone_label} 両唇閉鎖の視覚検証",
        fonts.body,
        700,
    )
    draw.text((276, 9), title_text, fill=WHITE, font=fonts.body)
    event_id_text = _ellipsize_text(draw, event.event_id, fonts.tiny, 700)
    draw.text((278, 43), event_id_text, fill=MUTED, font=fonts.tiny)

    finding_chip = (1008, 8, 1288, 64)
    finding_label = {
        "missing": "閉鎖確認できず",
        "reference": "同期参照" if event.reference_kind == "sync" else "閉鎖あり参照",
    }[event.category]
    finding_text = f"{finding_label} {category_order}/{category_count}"
    finding_text = _ellipsize_text(draw, finding_text, fonts.small, 246)
    draw.rounded_rectangle(finding_chip, radius=7, fill="#10161f", outline=event.color, width=3)
    finding_bbox = draw.textbbox((0, 0), finding_text, font=fonts.small)
    finding_width = finding_bbox[2] - finding_bbox[0]
    draw.text(
        (finding_chip[0] + (finding_chip[2] - finding_chip[0] - finding_width) // 2, 24),
        finding_text,
        fill=event.color,
        font=fonts.small,
    )

    speed_chip = (1304, 9, 1482, 63)
    speed_text = "1.0× 通常" if phase == "normal" else "0.25× スロー"
    draw.rounded_rectangle(speed_chip, radius=5, fill="#151c25", outline="#566170", width=2)
    speed_bbox = draw.textbbox((0, 0), speed_text, font=fonts.tiny)
    speed_width = speed_bbox[2] - speed_bbox[0]
    draw.text(
        (speed_chip[0] + (speed_chip[2] - speed_chip[0] - speed_width) // 2, 26),
        speed_text,
        fill=WHITE,
        font=fonts.tiny,
    )

    draw.text((1504, 27), MISSING_LEGEND, fill=MISSING_COLOR, font=fonts.tiny)
    red_bbox = draw.textbbox((1504, 27), MISSING_LEGEND, font=fonts.tiny)
    draw.text((red_bbox[2] + 17, 27), REFERENCE_LEGEND, fill=REFERENCE_COLOR, font=fonts.tiny)
    timestamp_text = f"原映像 {_source_timestamp(source_frame.time_s)}"
    timestamp_box = (34, HEADER_HEIGHT + 22, 334, HEADER_HEIGHT + 68)
    _rgba_overlay(canvas, timestamp_box, "#090d13", 205)
    draw = ImageDraw.Draw(canvas)
    draw.text((52, HEADER_HEIGHT + 31), timestamp_text, fill=WHITE, font=fonts.tiny)

    width, height = source_frame.image.size
    x0 = int(round(event.mouth_roi[0] * width))
    y0 = int(round(event.mouth_roi[1] * height))
    x1 = int(round(event.mouth_roi[2] * width))
    y1 = int(round(event.mouth_roi[3] * height))
    # ROI box on the source strip, clipped if the top/bottom edge lies outside.
    mapped = (
        int(round(x0 * crop_scale)),
        HEADER_HEIGHT + int(round((y0 - crop_top) * crop_scale)),
        int(round(x1 * crop_scale)),
        HEADER_HEIGHT + int(round((y1 - crop_top) * crop_scale)),
    )
    draw.rectangle(mapped, outline=event.color, width=4)

    inset_box = MOUTH_INSET_XYXY
    for start, end in _mouth_inset_connectors(mapped, inset_box):
        draw.line((start, end), fill="#080c12", width=7)
        draw.line((start, end), fill=WHITE, width=3)
    mouth = source_frame.image.crop((x0, y0, x1, y1))
    mouth_panel = _fit_crop(mouth, inset_box[2] - inset_box[0], inset_box[3] - inset_box[1])
    canvas.paste(mouth_panel, (inset_box[0], inset_box[1]))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(inset_box, outline=WHITE, width=5)
    label_box = (inset_box[0] + 12, inset_box[1] + 12, inset_box[0] + 164, inset_box[1] + 51)
    draw.rounded_rectangle(label_box, radius=9, fill="#070b11")
    draw.text((label_box[0] + 15, label_box[1] + 8), "口元拡大", fill=WHITE, font=fonts.tiny)

    info_box = (1396, HEADER_HEIGHT + 364, 1882, HEADER_HEIGHT + 608)
    _rgba_overlay(canvas, info_box, "#080c12", 225)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(info_box, radius=16, outline=event.color, width=3)
    finding_font = fonts.finding if event.category == "missing" else fonts.header
    draw.text((1424, HEADER_HEIGHT + 386), event.category_ja, fill=event.color, font=finding_font)
    display_label = _ellipsize_text(draw, event.label, fonts.body, 420)
    draw.text((1424, HEADER_HEIGHT + 438), display_label, fill=WHITE, font=fonts.body)
    note_lines = _wrap_text(draw, event.note, fonts.tiny, 425)
    for line_index, line in enumerate(note_lines[:3]):
        draw.text((1424, HEADER_HEIGHT + 486 + 25 * line_index), line, fill=MUTED, font=fonts.tiny)
    phase_ja = "通常速度 1.0×" if phase == "normal" else "スロー 0.25×"
    draw.rounded_rectangle((1424, HEADER_HEIGHT + 556, 1658, HEADER_HEIGHT + 594), radius=9, fill=event.color)
    draw.text((1441, HEADER_HEIGHT + 564), phase_ja, fill="#071008", font=fonts.tiny)

    if event.warning:
        warning_box = (34, HEADER_HEIGHT + VIDEO_HEIGHT - 70, 1070, HEADER_HEIGHT + VIDEO_HEIGHT - 20)
        _rgba_overlay(canvas, warning_box, "#080c12", 230)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(warning_box, radius=12, outline="#f5c451", width=2)
        warning_text = _ellipsize_text(
            draw,
            f"注意：{event.warning}",
            fonts.tiny,
            warning_box[2] - warning_box[0] - 40,
        )
        draw.text((54, warning_box[1] + 13), warning_text, fill="#f5c451", font=fonts.tiny)

    _draw_waveform(canvas, event, envelope, source_frame.time_s, phase, fonts, phone_label)
    return canvas.convert("RGB")


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        proposed = current + character
        width = draw.textbbox((0, 0), proposed, font=font)[2]
        if current and width > max_width:
            lines.append(current)
            current = character
        else:
            current = proposed
    if current:
        lines.append(current)
    return lines


def render_card(
    manifest: Manifest,
    fonts: FontSet,
    *,
    outro: bool,
    missing_count: int,
    reference_count: int,
) -> Image.Image:
    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), PANEL_2)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 18, OUTPUT_HEIGHT), fill=MISSING_COLOR)
    draw.rectangle((18, 0, 30, OUTPUT_HEIGHT), fill=REFERENCE_COLOR)
    heading = "比較映像 終了" if outro else manifest.title
    draw.text((132, 220), heading, fill=WHITE, font=fonts.huge)
    draw.text((136, 316), manifest.subtitle, fill=MUTED, font=fonts.title)
    draw.rounded_rectangle((136, 426, 616, 514), radius=20, outline=MISSING_COLOR, width=4)
    draw.text((171, 450), f"閉鎖確認できず　{missing_count}件", fill=MISSING_COLOR, font=fonts.body)
    draw.rounded_rectangle((652, 426, 1132, 514), radius=20, outline=REFERENCE_COLOR, width=4)
    draw.text((687, 450), f"閉鎖あり参照　{reference_count}件", fill=REFERENCE_COLOR, font=fonts.body)
    draw.line((136, 608, 1784, 608), fill=GRID, width=2)
    limitation_lines = _wrap_text(draw, CAUSAL_LIMITATION, fonts.body, 1550)
    for index, line in enumerate(limitation_lines):
        draw.text((136, 660 + index * 44), line, fill=WHITE, font=fonts.body)
    draw.text((136, 824), "赤＝明瞭な閉鎖を確認できず　　緑＝閉鎖あり参照　　水色＝再生位置", fill=MUTED, font=fonts.small)
    draw.text((136, 904), "通常：原フレーム1:1　／　スロー：各原フレームを4回保持（補間なし）", fill=MUTED, font=fonts.small)
    return canvas


class AVEncoder:
    def __init__(self, path: Path, *, crf: int, preset: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.video = self.container.add_stream("libx264", rate=OUTPUT_FPS)
        self.video.width = OUTPUT_WIDTH
        self.video.height = OUTPUT_HEIGHT
        self.video.pix_fmt = "yuv420p"
        self.video.options = {"preset": preset, "crf": str(crf), "tune": "zerolatency"}
        self.audio = self.container.add_stream("aac", rate=AUDIO_RATE)
        self.audio.layout = "stereo"
        self.audio.bit_rate = 192_000
        self.frame_index = 0
        self.audio_pts = 0
        self._pending_audio = np.zeros((2, 0), dtype=np.float32)

    def add(self, image: Image.Image, audio: np.ndarray) -> None:
        if audio.shape != (2, AUDIO_SAMPLES_PER_VIDEO_FRAME):
            raise ValueError(f"audio block must be (2,{AUDIO_SAMPLES_PER_VIDEO_FRAME}), got {audio.shape}")
        video_frame = av.VideoFrame.from_image(image)
        video_frame.pts = self.frame_index
        video_frame.time_base = Fraction(1, OUTPUT_FPS)
        for packet in self.video.encode(video_frame):
            self.container.mux(packet)
        self.frame_index += 1
        self._pending_audio = np.concatenate((self._pending_audio, audio.astype(np.float32, copy=False)), axis=1)
        while self._pending_audio.shape[1] >= 1024:
            block = np.ascontiguousarray(self._pending_audio[:, :1024])
            self._pending_audio = self._pending_audio[:, 1024:]
            self._encode_audio_block(block)

    def _encode_audio_block(self, block: np.ndarray) -> None:
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
        frame.sample_rate = AUDIO_RATE
        frame.pts = self.audio_pts
        frame.time_base = Fraction(1, AUDIO_RATE)
        self.audio_pts += block.shape[1]
        for packet in self.audio.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self._pending_audio.shape[1]:
            pad = 1024 - self._pending_audio.shape[1]
            block = np.pad(self._pending_audio, ((0, 0), (0, pad))).astype(np.float32)
            self._encode_audio_block(block)
            self._pending_audio = np.zeros((2, 0), dtype=np.float32)
        for packet in self.video.encode():
            self.container.mux(packet)
        for packet in self.audio.encode():
            self.container.mux(packet)
        self.container.close()


FRAME_MAP_FIELDS = (
    "output_frame_index",
    "output_pts_s",
    "case_order",
    "event_id",
    "phase",
    "source_pts_s",
    "source_frame_index",
)


def _silence(frame_count: int) -> np.ndarray:
    return np.zeros((2, frame_count * AUDIO_SAMPLES_PER_VIDEO_FRAME), dtype=np.float32)


def _audio_block(audio: np.ndarray, frame_index: int) -> np.ndarray:
    start = frame_index * AUDIO_SAMPLES_PER_VIDEO_FRAME
    end = start + AUDIO_SAMPLES_PER_VIDEO_FRAME
    return np.ascontiguousarray(audio[:, start:end])


def render_video(
    video_path: Path,
    manifest: Manifest,
    output_path: Path,
    frame_map_path: Path,
    *,
    intro_seconds: float,
    outro_seconds: float,
    event_gap_seconds: float,
    crf: int,
    preset: str,
    quiet: bool,
) -> None:
    fonts = make_fonts()
    missing_count = sum(event.category == "missing" for event in manifest.events)
    reference_count = sum(event.category == "reference" for event in manifest.events)
    category_totals = {"missing": missing_count, "reference": reference_count}
    category_seen = {"missing": 0, "reference": 0}
    encoder = AVEncoder(output_path, crf=crf, preset=preset)
    frame_map_path.parent.mkdir(parents=True, exist_ok=True)
    with frame_map_path.open("w", encoding="utf-8", newline="") as handle:
        frame_map = csv.DictWriter(handle, fieldnames=FRAME_MAP_FIELDS)
        frame_map.writeheader()

        def emit(
            image: Image.Image,
            audio_block: np.ndarray,
            *,
            case_order: int | str,
            event_id: str,
            phase: str,
            source: SourceFrame | None,
        ) -> None:
            output_index = encoder.frame_index
            encoder.add(image, audio_block)
            frame_map.writerow(
                {
                    "output_frame_index": output_index,
                    "output_pts_s": f"{output_index / OUTPUT_FPS:.9f}",
                    "case_order": case_order,
                    "event_id": event_id,
                    "phase": phase,
                    "source_pts_s": "" if source is None else f"{source.time_s:.9f}",
                    "source_frame_index": "" if source is None else source.source_frame_index,
                }
            )

        intro_frames = max(0, int(round(intro_seconds * OUTPUT_FPS)))
        intro_card = render_card(
            manifest,
            fonts,
            outro=False,
            missing_count=missing_count,
            reference_count=reference_count,
        )
        for index in range(intro_frames):
            emit(
                intro_card,
                _audio_block(_silence(intro_frames), index),
                case_order="",
                event_id="",
                phase="intro",
                source=None,
            )

        for case_index, event in enumerate(manifest.events, start=1):
            category_seen[event.category] += 1
            if not quiet:
                print(
                    f"[{case_index}/{len(manifest.events)}] {event.event_id} "
                    f"{event.category_ja} @ {event.release_s:.3f}s",
                    flush=True,
                )
            decoded, _ = decode_video_window(video_path, event)
            native_frames = select_native_frames(decoded, event)
            source_audio = decode_audio_window(video_path, event)
            envelope = waveform_envelope(source_audio, OUTPUT_WIDTH - 144)
            normal_audio = stretch_audio(
                source_audio,
                len(native_frames) * AUDIO_SAMPLES_PER_VIDEO_FRAME,
            )
            for local_index, source in enumerate(native_frames):
                image = render_evidence_frame(
                    source,
                    event,
                    envelope,
                    "normal",
                    case_index,
                    len(manifest.events),
                    category_seen[event.category],
                    category_totals[event.category],
                    fonts,
                    manifest.phone_label,
                )
                emit(
                    image,
                    _audio_block(normal_audio, local_index),
                    case_order=case_index,
                    event_id=event.event_id,
                    phase="normal",
                    source=source,
                )

            slow_frames = [source for source in native_frames for _ in range(4)]
            slow_audio = stretch_audio(
                source_audio,
                len(slow_frames) * AUDIO_SAMPLES_PER_VIDEO_FRAME,
            )
            for local_index, source in enumerate(slow_frames):
                image = render_evidence_frame(
                    source,
                    event,
                    envelope,
                    "slow",
                    case_index,
                    len(manifest.events),
                    category_seen[event.category],
                    category_totals[event.category],
                    fonts,
                    manifest.phone_label,
                )
                emit(
                    image,
                    _audio_block(slow_audio, local_index),
                    case_order=case_index,
                    event_id=event.event_id,
                    phase="slow",
                    source=source,
                )

            gap_frames = max(0, int(round(event_gap_seconds * OUTPUT_FPS)))
            if gap_frames:
                hold = render_evidence_frame(
                    native_frames[-1],
                    event,
                    envelope,
                    "slow",
                    case_index,
                    len(manifest.events),
                    category_seen[event.category],
                    category_totals[event.category],
                    fonts,
                    manifest.phone_label,
                )
                gap_audio = _silence(gap_frames)
                for local_index in range(gap_frames):
                    emit(
                        hold,
                        _audio_block(gap_audio, local_index),
                        case_order=case_index,
                        event_id=event.event_id,
                        phase="gap",
                        source=native_frames[-1],
                    )

        outro_frames = max(0, int(round(outro_seconds * OUTPUT_FPS)))
        outro_card = render_card(
            manifest,
            fonts,
            outro=True,
            missing_count=missing_count,
            reference_count=reference_count,
        )
        outro_audio = _silence(outro_frames)
        for index in range(outro_frames):
            emit(
                outro_card,
                _audio_block(outro_audio, index),
                case_order="",
                event_id="",
                phase="outro",
                source=None,
            )
    encoder.close()


def write_effective_manifest(
    path: Path,
    *,
    input_manifest: Path,
    source_video: Path,
    output_video: Path,
    frame_map: Path,
    manifest: Manifest,
    requested_pre_s: float,
    requested_post_s: float,
    intro_seconds: float,
    outro_seconds: float,
    event_gap_seconds: float,
    source_frame_timing: dict[str, object] | None = None,
    approved_layout_review: dict[str, Any],
) -> None:
    """Record the effective, uniform render parameters separately from selection data."""
    intro_frames = max(0, int(round(intro_seconds * OUTPUT_FPS)))
    outro_frames = max(0, int(round(outro_seconds * OUTPUT_FPS)))
    gap_frames = max(0, int(round(event_gap_seconds * OUTPUT_FPS)))
    cursor = intro_frames
    cases: list[dict[str, Any]] = []
    for index, event in enumerate(manifest.events, start=1):
        normal_frames = int(round((event.pre_s + event.post_s) * OUTPUT_FPS))
        slow_frames = normal_frames * 4
        normal_start_frame = cursor
        normal_end_frame = normal_start_frame + normal_frames
        slow_start_frame = normal_end_frame
        slow_end_frame = slow_start_frame + slow_frames
        classification = event.classification
        cases.append(
            {
                "order": index,
                "event_id": event.event_id,
                "classification": classification,
                "presentation_label": event.category_ja,
                "label": event.label,
                "note": event.note,
                "warning": event.warning,
                "source_release_s": event.release_s,
                "mouth_crop_normalized_xyxy": list(event.mouth_roi),
                "mouth_crop_xywh": [
                    round(event.mouth_roi[0] * OUTPUT_WIDTH),
                    round(event.mouth_roi[1] * OUTPUT_HEIGHT),
                    round((event.mouth_roi[2] - event.mouth_roi[0]) * OUTPUT_WIDTH),
                    round((event.mouth_roi[3] - event.mouth_roi[1]) * OUTPUT_HEIGHT),
                ],
                "normal": {
                    "source_start_s": event.start_s,
                    "source_end_s": event.end_s,
                    "output_start_s": normal_start_frame / OUTPUT_FPS,
                    "output_end_s": normal_end_frame / OUTPUT_FPS,
                    "speed": 1.0,
                    "marker_output_s": normal_start_frame / OUTPUT_FPS + event.pre_s,
                    "output_frame_count": normal_frames,
                },
                "slow": {
                    "source_start_s": event.start_s,
                    "source_end_s": event.end_s,
                    "output_start_s": slow_start_frame / OUTPUT_FPS,
                    "output_end_s": slow_end_frame / OUTPUT_FPS,
                    "speed": 0.25,
                    "marker_output_s": slow_start_frame / OUTPUT_FPS + event.pre_s / 0.25,
                    "output_frame_count": slow_frames,
                },
            }
        )
        cursor = slow_end_frame + gap_frames
    total_frames = cursor + outro_frames
    payload = {
        "schema_version": 1,
        "purpose": "visual comparison of a bilabial-plosive release and bilabial contact",
        "phone_label": manifest.phone_label,
        "input_event_manifest": str(input_manifest.resolve()),
        "source_video": str(source_video.resolve()),
        "output_video": str(output_video.resolve()),
        "frame_map": str(frame_map.resolve()),
        "render": {
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "fps": OUTPUT_FPS,
            "audio_sample_rate": AUDIO_RATE,
            "normal_source_frames_per_output_frame": 1,
            "slow_source_frame_repeat": 4,
            "video_interpolation": False,
            "source_frame_timing": source_frame_timing,
            "requested_uniform_pre_release_s": requested_pre_s,
            "requested_uniform_post_release_s": requested_post_s,
            "window_quantization_note": (
                "CFR 24 fps and normal 1:1 source mapping require whole-frame "
                "windows; effective per-event values are recorded below."
            ),
            "limitation": CAUSAL_LIMITATION,
            "slow_audio_processing": (
                "linear time-domain resampling to 4x duration; pitch is not preserved; "
                "the same source window is used for audio and video"
            ),
            "waveform": {
                "source": "mean of the two decoded source channels",
                "feature": "per-horizontal-pixel RMS envelope",
                "normalization": "each case divided by its 99th-percentile RMS; same rule for all cases",
                "comparison_limitation": NORMALIZATION_LIMITATION,
                "clip_range": [0.0, 1.0],
                "release_marker": "fixed manifest source_release_s; not peak-picked for display",
            },
            "mouth_crop": {
                "mode": "static normalized ROI for the candidate's stable full-screen framing",
                "tracking": False,
                "spatial_interpolation": "Pillow bilinear resize only",
                "temporal_interpolation": False,
            },
            "layout": {
                "header_height_px": HEADER_HEIGHT,
                "source_video_height_px": VIDEO_HEIGHT,
                "waveform_top_px": WAVEFORM_Y,
                "mouth_inset_xyxy": list(MOUTH_INSET_XYXY),
            },
            "layout_reference": load_approved_layout_reference(),
            "layout_review": approved_layout_review,
        },
        "cards": {
            "intro_frames": intro_frames,
            "intro_duration_s": intro_frames / OUTPUT_FPS,
            "event_gap_frames": gap_frames,
            "event_gap_duration_s": gap_frames / OUTPUT_FPS,
            "outro_frames": outro_frames,
            "outro_duration_s": outro_frames / OUTPUT_FPS,
        },
        "classification_counts": {
            "closure_absent": sum(event.category == "missing" for event in manifest.events),
            "contact_reference": sum(
                event.category == "reference" and event.reference_kind == "contact"
                for event in manifest.events
            ),
            "sync_reference": sum(
                event.category == "reference" and event.reference_kind == "sync"
                for event in manifest.events
            ),
        },
        "cases": cases,
        "output": {
            "total_frame_count": total_frames,
            "expected_duration_s": total_frames / OUTPUT_FPS,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="source MP4; overrides manifest source_video")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, help="default: closure_evidence_frame_map.csv beside output")
    parser.add_argument(
        "--effective-manifest",
        type=Path,
        help="default: closure_evidence_manifest.json beside output",
    )
    parser.add_argument("--limit-events", type=int, help="render only the first N events (smoke/QA)")
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=0.8,
        help="uniform pre-release render window (default: 0.8)",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=0.8,
        help="uniform post-release render window (default: 0.8)",
    )
    parser.add_argument("--intro-seconds", type=float, default=5.0)
    parser.add_argument("--outro-seconds", type=float, default=5.0)
    parser.add_argument("--event-gap-seconds", type=float, default=0.25)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        layout_reference = load_approved_layout_reference()
        approved_layout_review = verify_approved_preview(
            args.manifest.resolve(), layout_reference
        )
    except Exception as error:
        print(
            f"ERROR: approved layout/preview gate failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    manifest = load_manifest(args.manifest.resolve())
    video_path = (args.video or manifest.source_video)
    if video_path is None:
        raise SystemExit("source video is required through --video or source_video in manifest")
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"source video not found: {video_path}")
    events = list(manifest.events)
    if args.pre_seconds <= 0:
        raise SystemExit("--pre-seconds must be positive")
    events = [replace(event, pre_s=args.pre_seconds) for event in events]
    if args.post_seconds <= 0:
        raise SystemExit("--post-seconds must be positive")
    events = [replace(event, post_s=args.post_seconds) for event in events]
    events = [quantize_event_window(event) for event in events]
    if args.limit_events is not None:
        if args.limit_events <= 0:
            raise SystemExit("--limit-events must be positive")
        events = events[: args.limit_events]
    manifest = replace(manifest, events=tuple(events))
    frame_map = args.frame_map or (args.output.parent / "closure_evidence_frame_map.csv")
    effective_manifest = args.effective_manifest or (
        args.output.parent / "closure_evidence_manifest.json"
    )
    try:
        frame_timing = probe_video_frame_timing(video_path, expected_fps=OUTPUT_FPS)
        if not frame_timing.is_cfr:
            raise RuntimeError(
                "source must have strict 24fps CFR presentation timestamps; "
                f"diagnostics: {frame_timing.as_dict()}"
            )
        render_video(
            video_path,
            manifest,
            args.output.resolve(),
            frame_map.resolve(),
            intro_seconds=max(0.0, args.intro_seconds),
            outro_seconds=max(0.0, args.outro_seconds),
            event_gap_seconds=max(0.0, args.event_gap_seconds),
            crf=args.crf,
            preset=args.preset,
            quiet=args.quiet,
        )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    write_effective_manifest(
        effective_manifest.resolve(),
        input_manifest=args.manifest,
        source_video=video_path,
        output_video=args.output,
        frame_map=frame_map,
        manifest=manifest,
        requested_pre_s=args.pre_seconds,
        requested_post_s=args.post_seconds,
        intro_seconds=max(0.0, args.intro_seconds),
        outro_seconds=max(0.0, args.outro_seconds),
        event_gap_seconds=max(0.0, args.event_gap_seconds),
        source_frame_timing=frame_timing.as_dict(),
        approved_layout_review=approved_layout_review,
    )
    if not args.quiet:
        print(f"Rendered: {args.output.resolve()}")
        print(f"Frame map: {frame_map.resolve()}")
        print(f"Effective manifest: {effective_manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
