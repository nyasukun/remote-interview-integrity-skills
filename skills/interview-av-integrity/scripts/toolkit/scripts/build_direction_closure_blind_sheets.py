#!/usr/bin/env python3
"""Build speaker- and classifier-blinded visual-contact audit strips.

Eligible events are shuffled with a fixed seed.  The rendered strips omit the
event ID, speaker, token, and machine classification; the separate key retains
the stable ``VC-NNN`` mapping needed for unblinding.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import av
from PIL import Image, ImageDraw, ImageFont


SEED = 1729
DEFAULT_WINDOW_BEFORE_S = 0.250
DEFAULT_WINDOW_AFTER_S = 0.167
DEFAULT_ROI = (0.30, 0.34, 0.70, 0.96)
DEFAULT_COLUMNS = 6
REVIEW_COLUMN_ALIASES = {
    "contact": (
        "contact",
        "human_contact_full_window",
        "display_window_clear_contact",
        "clear_lip_contact",
    ),
    "timing_relative_audio": (
        "timing_relative_audio",
        "first_contact_timing",
    ),
    "machine_window_category": (
        "machine_window_category",
        "human_machine_window_category",
        "machine_window_contact",
        "contact_strength_250ms_to_83ms",
    ),
}


def normalized_review_value(
    row: dict[str, str], canonical_name: str, *, required: bool = True
) -> str:
    """Read a canonical review field while accepting documented legacy headings."""

    aliases = REVIEW_COLUMN_ALIASES.get(canonical_name, (canonical_name,))
    populated = {
        str(row.get(column, "")).strip().lower()
        for column in aliases
        if str(row.get(column, "")).strip()
    }
    if len(populated) > 1:
        raise RuntimeError(
            f"Conflicting values for {canonical_name}: {sorted(populated)}"
        )
    if populated:
        return populated.pop()
    if required:
        raise RuntimeError(
            f"Missing {canonical_name}; accepted columns: {', '.join(aliases)}"
        )
    return ""


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def blinded_events(
    payload: dict,
    *,
    phoneme_class: str = "p",
    seed: int = SEED,
    expected_count: int | None = None,
) -> list[dict]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise RuntimeError("Events payload must contain an events list")
    selected = [
        event
        for event in raw_events
        if event.get("selection", {}).get("phoneme_class") == phoneme_class
        and event.get("analysis_eligible") is True
        and event.get("geometry", {}).get("face_valid") is True
    ]
    event_ids = [str(event.get("runner_event_id", "")).strip() for event in selected]
    if any(not event_id for event_id in event_ids):
        raise RuntimeError("Selected events must have nonempty runner_event_id values")
    if len(set(event_ids)) != len(event_ids):
        raise RuntimeError("Selected events must have unique runner_event_id values")
    selected.sort(key=lambda event: event["runner_event_id"])
    random.Random(seed).shuffle(selected)
    if not selected:
        raise RuntimeError(
            f"No face-valid, analysis-eligible /{phoneme_class}/ events found"
        )
    if expected_count is not None and len(selected) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} eligible /{phoneme_class}/ events, "
            f"found {len(selected)}"
        )
    return selected


def decode_window(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    release_s: float,
    *,
    window_before_s: float = DEFAULT_WINDOW_BEFORE_S,
    window_after_s: float = DEFAULT_WINDOW_AFTER_S,
) -> list[tuple[float, Image.Image]]:
    start_s = release_s - window_before_s
    end_s = release_s + window_after_s
    container.seek(
        int(max(0.0, start_s - 1.0) / float(stream.time_base)),
        stream=stream,
        backward=True,
    )
    frames: list[tuple[float, Image.Image]] = []
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        time_s = float(frame.pts * frame.time_base)
        if time_s < start_s:
            continue
        if time_s > end_s:
            break
        frames.append((time_s, frame.to_image()))
    if not frames:
        raise RuntimeError(f"No native frames found around {release_s:.3f}s")
    return frames


def validate_roi(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("ROI needs four normalized values: x0 y0 x1 y1")
    x0, y0, x1, y1 = (float(value) for value in values)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def normalized_roi_crop(
    image: Image.Image, roi: Sequence[float] = DEFAULT_ROI
) -> Image.Image:
    x0, y0, x1, y1 = validate_roi(roi)
    width, height = image.size
    return image.crop(
        (
            int(round(width * x0)),
            int(round(height * y0)),
            int(round(width * x1)),
            int(round(height * y1)),
        )
    )


def stream_fps(stream: av.video.stream.VideoStream) -> float:
    for value in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        if value is not None:
            result = float(value)
            if math.isfinite(result) and result > 0:
                return result
    raise RuntimeError("Unable to determine a positive native video frame rate")


def render_strip(
    blind_id: str,
    release_s: float,
    frames: list[tuple[float, Image.Image]],
    *,
    fps: float,
    roi: Sequence[float] = DEFAULT_ROI,
    columns: int = DEFAULT_COLUMNS,
) -> Image.Image:
    if not frames:
        raise ValueError("Cannot render an empty frame list")
    if columns <= 0:
        raise ValueError("columns must be positive")
    panel_width = 310
    panel_height = 300
    rows = math.ceil(len(frames) / columns)
    header_height = 58
    canvas = Image.new("RGB", (columns * panel_width, header_height + rows * panel_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24, bold=True)
    label_font = _font(17)
    draw.text((14, 8), blind_id, fill="black", font=title_font)
    draw.text(
        (140, 13),
        f"speaker/classifier hidden | native {fps:.3f} fps | red = nearest frame to audio release",
        fill="#333333",
        font=label_font,
    )
    nearest_index = min(
        range(len(frames)), key=lambda index: abs(frames[index][0] - release_s)
    )
    for index, (frame_time_s, image) in enumerate(frames):
        row, column = divmod(index, columns)
        x0 = column * panel_width
        y0 = header_height + row * panel_height
        crop = normalized_roi_crop(image, roi)
        crop.thumbnail((panel_width - 12, panel_height - 42), Image.Resampling.LANCZOS)
        x = x0 + (panel_width - crop.width) // 2
        y = y0 + 4
        canvas.paste(crop, (x, y))
        is_nearest = index == nearest_index
        draw.rectangle(
            (x, y, x + crop.width - 1, y + crop.height - 1),
            outline="#d62728" if is_nearest else "#777777",
            width=5 if is_nearest else 1,
        )
        delta_ms = (frame_time_s - release_s) * 1000.0
        draw.text(
            (x0 + 90, y0 + panel_height - 31),
            f"{delta_ms:+.1f} ms",
            fill="#b2182b" if is_nearest else "#222222",
            font=label_font,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--phoneme-class", default="p")
    parser.add_argument("--window-before-s", type=float, default=DEFAULT_WINDOW_BEFORE_S)
    parser.add_argument("--window-after-s", type=float, default=DEFAULT_WINDOW_AFTER_S)
    parser.add_argument("--roi", nargs=4, type=float, default=DEFAULT_ROI, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument(
        "--expected-event-count",
        type=int,
        help="Optional fail-closed count check; by default any nonempty count is accepted.",
    )
    args = parser.parse_args()

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    if (
        not math.isfinite(args.window_before_s)
        or not math.isfinite(args.window_after_s)
        or args.window_before_s <= 0
        or args.window_after_s <= 0
    ):
        parser.error("window values must be positive")
    if args.columns <= 0:
        parser.error("--columns must be positive")
    if args.expected_event_count is not None and args.expected_event_count <= 0:
        parser.error("--expected-event-count must be positive")
    roi = validate_roi(args.roi)
    events = blinded_events(
        payload,
        phoneme_class=args.phoneme_class,
        seed=SEED,
        expected_count=args.expected_event_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = stream_fps(stream)
    key: list[dict] = []
    try:
        for index, event in enumerate(events, start=1):
            blind_id = f"VC-{index:03d}"
            release_s = float(event["audio_release_time_s"])
            frames = decode_window(
                container,
                stream,
                release_s,
                window_before_s=args.window_before_s,
                window_after_s=args.window_after_s,
            )
            strip = render_strip(
                blind_id,
                release_s,
                frames,
                fps=fps,
                roi=roi,
                columns=args.columns,
            )
            strip.save(args.output_dir / f"{blind_id}.png", optimize=True)
            key.append(
                {
                    "blind_id": blind_id,
                    "runner_event_id": event["runner_event_id"],
                    "audio_release_time_s": release_s,
                    "native_frame_offsets_ms": [
                        round((time_s - release_s) * 1000.0, 6)
                        for time_s, _ in frames
                    ],
                }
            )
    finally:
        container.close()
    args.blind_key.parent.mkdir(parents=True, exist_ok=True)
    args.blind_key.write_text(
        json.dumps(
            {
                "seed": SEED,
                "phoneme_class": args.phoneme_class,
                "event_count": len(key),
                "native_fps": fps,
                "window_before_ms": args.window_before_s * 1000.0,
                "window_after_ms": args.window_after_s * 1000.0,
                "normalized_roi_xyxy": list(roi),
                "columns": args.columns,
                "events": key,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(key)} blinded strips to {args.output_dir}")


if __name__ == "__main__":
    main()
