#!/usr/bin/env python3
"""Render configurable blinded contact follow-up strips."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import av
from PIL import Image, ImageDraw

from build_direction_closure_blind_sheets import (
    DEFAULT_ROI,
    _font,
    normalized_review_value,
    normalized_roi_crop,
    stream_fps,
    validate_roi,
)


DEFAULT_START_AFTER_S = 0.125
DEFAULT_END_AFTER_S = 0.500
DEFAULT_COLUMNS = 5


def decode_followup(
    container,
    stream,
    release_s: float,
    *,
    start_after_s: float = DEFAULT_START_AFTER_S,
    end_after_s: float = DEFAULT_END_AFTER_S,
) -> list[tuple[float, Image.Image]]:
    start_s = release_s + start_after_s
    end_s = release_s + end_after_s
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
        raise RuntimeError(f"No follow-up frames found at {release_s:.3f}s")
    return frames


def render(
    blind_id: str,
    release_s: float,
    frames: list[tuple[float, Image.Image]],
    *,
    fps: float,
    start_after_s: float,
    end_after_s: float,
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
        f"follow-up hidden review | native {fps:.3f} fps | "
        f"{start_after_s * 1000:+.0f} through {end_after_s * 1000:+.0f} ms",
        fill="#333333",
        font=label_font,
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
        draw.rectangle((x, y, x + crop.width - 1, y + crop.height - 1), outline="#777777", width=1)
        delta_ms = (frame_time_s - release_s) * 1000.0
        draw.text((x0 + 90, y0 + panel_height - 31), f"{delta_ms:+.1f} ms", fill="#222222", font=label_font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-after-s", type=float, default=DEFAULT_START_AFTER_S)
    parser.add_argument("--end-after-s", type=float, default=DEFAULT_END_AFTER_S)
    parser.add_argument("--roi", nargs=4, type=float, default=DEFAULT_ROI, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument(
        "--expected-event-count",
        type=int,
        help="Optional fail-closed count check; by default any nonempty count is accepted.",
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.start_after_s)
        or not math.isfinite(args.end_after_s)
        or args.start_after_s < 0
        or args.end_after_s <= args.start_after_s
    ):
        parser.error("follow-up window must satisfy 0 <= start < end")
    if args.columns <= 0:
        parser.error("--columns must be positive")
    if args.expected_event_count is not None and args.expected_event_count <= 0:
        parser.error("--expected-event-count must be positive")
    roi = validate_roi(args.roi)
    key_rows = json.loads(args.blind_key.read_text(encoding="utf-8"))["events"]
    key = {row["blind_id"]: row for row in key_rows}
    if not key or len(key) != len(key_rows):
        raise RuntimeError("Blind key must contain unique, nonempty blind IDs")
    with args.annotations.open(encoding="utf-8", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    annotation_ids = [str(row.get("blind_id", "")).strip() for row in annotations]
    if any(not blind_id for blind_id in annotation_ids):
        raise RuntimeError("Annotations must contain nonempty blind_id values")
    if len(set(annotation_ids)) != len(annotation_ids):
        raise RuntimeError("Annotations must contain unique blind_id values")
    selected = [
        row
        for row in annotations
        if normalized_review_value(row, "contact") == "no"
        or normalized_review_value(row, "timing_relative_audio")
        in {"after_only", "ambiguous"}
    ]
    if not selected:
        raise RuntimeError("No annotations met the configured follow-up rule")
    if (
        args.expected_event_count is not None
        and len(selected) != args.expected_event_count
    ):
        raise RuntimeError(
            f"Expected {args.expected_event_count} follow-up events, "
            f"found {len(selected)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = stream_fps(stream)
    try:
        for row in selected:
            blind_id = row["blind_id"]
            if blind_id not in key:
                raise RuntimeError(f"Annotation blind_id missing from key: {blind_id}")
            release_s = float(key[blind_id]["audio_release_time_s"])
            frames = decode_followup(
                container,
                stream,
                release_s,
                start_after_s=args.start_after_s,
                end_after_s=args.end_after_s,
            )
            render(
                blind_id,
                release_s,
                frames,
                fps=fps,
                start_after_s=args.start_after_s,
                end_after_s=args.end_after_s,
                roi=roi,
                columns=args.columns,
            ).save(
                args.output_dir / f"{blind_id}_followup.png", optimize=True
            )
    finally:
        container.close()
    print(f"Rendered {len(selected)} follow-up strips")


if __name__ == "__main__":
    main()
