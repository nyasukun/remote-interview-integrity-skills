#!/usr/bin/env python3
"""Render a local layout preview with synthetic waveform/Log-Mel placeholders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_DIR = SCRIPT_DIR / "toolkit"
RENDERING_DIR = TOOLKIT_DIR / "rendering"
for directory in (RENDERING_DIR, TOOLKIT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from acoustics.verify_artifacts import verify_acoustic_dir  # noqa: E402
from adapter import display_metrics  # noqa: E402
from common import font, load_manifest  # noqa: E402
from layout_reference import (  # noqa: E402
    preview_provenance_path,
    review_sheet_path,
    verify_layout_reference,
    write_preview_provenance,
)
from render import ClipData, _clip_frame, _summary  # noqa: E402


def _placeholder_wave(index: int, count: int = 820) -> np.ndarray:
    x = np.linspace(0.0, 1.0, count, endpoint=False)
    envelope = 0.22 + 0.18 * np.sin(2.0 * np.pi * (2 + index) * x) ** 2
    center = 0.08 * np.sin(2.0 * np.pi * (5 + index) * x)
    return np.column_stack((center - envelope, center + envelope)).astype(np.float32)


def _placeholder_spectrogram(
    color: tuple[int, int, int], index: int, width: int = 820, height: int = 240
) -> Image.Image:
    x = np.linspace(0.0, 1.0, width, endpoint=False)[None, :]
    y = np.linspace(0.0, 1.0, height, endpoint=False)[:, None]
    bands = (
        np.exp(-((y - (0.18 + 0.03 * index)) ** 2) / 0.006)
        + 0.7 * np.exp(-((y - 0.43) ** 2) / 0.012)
        + 0.35 * np.exp(-((y - 0.70) ** 2) / 0.020)
    )
    modulation = 0.45 + 0.55 * np.sin(2.0 * np.pi * (3 + index) * x) ** 2
    strength = np.clip(bands * modulation, 0.0, 1.0)
    base = np.asarray((3, 9, 16), dtype=np.float64)
    tint = np.asarray(color, dtype=np.float64)
    rgb = base + strength[..., None] * tint
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def render_preview(manifest: Path, output: Path) -> Path:
    # Fail before reading case data if the approved composition reference has
    # been removed, replaced, or changed.
    reference = verify_layout_reference()
    spec = load_manifest(manifest)
    artifact_manifest = Path(spec.acoustic_artifacts["artifact_manifest"]["path"])
    acoustic_features = Path(spec.acoustic_artifacts["acoustic_features"]["path"])
    if artifact_manifest.parent != acoustic_features.parent:
        raise ValueError("authoritative acoustic artifacts must share one directory")
    verification = verify_acoustic_dir(artifact_manifest.parent)
    if verification.get("status") != "PASS":
        raise ValueError("authoritative acoustic artifacts did not pass verification")
    payload = json.loads(acoustic_features.read_text(encoding="utf-8"))
    authoritative = {
        str(row.get("clip_id", "")): row for row in payload.get("clips", [])
    }
    if set(authoritative) != {clip.clip_id for clip in spec.clips}:
        raise ValueError("preview clip IDs differ from authoritative acoustic artifacts")

    group_by_id = {group.group_id: group for group in spec.groups}
    clips: list[ClipData] = []
    for index, clip in enumerate(spec.clips):
        row = authoritative[clip.clip_id]
        if row.get("group_id") != clip.group_id:
            raise ValueError(f"{clip.clip_id}: authoritative group mismatch")
        metrics = {
            **display_metrics(row["summary"]),
            "authoritative_summary": row["summary"],
        }
        data = ClipData(
            spec=clip,
            audio=np.zeros((2, 1), dtype=np.float32),
            metrics=metrics,
            decode={"preview": True},
            waveform=_placeholder_wave(index),
            spectrogram=_placeholder_spectrogram(
                group_by_id[clip.group_id].color_rgb, index
            ),
        )
        data.output_start_sample = index * 96_000
        data.output_end_sample = data.output_start_sample + 48_000
        clips.append(data)

    summary = _summary(spec, clips)
    first_reference = next(
        clip for clip in clips if clip.spec.group_id != spec.anchor_group
    )
    sample = first_reference.output_start_sample + 24_000
    image = _clip_frame(spec, clips, summary, sample)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (18, 9, 535, 43), radius=8, fill=(25, 39, 50), outline=(105, 122, 135)
    )
    draw.text(
        (32, 17),
        "LAYOUT PREVIEW — waveform / Log-Mel are synthetic placeholders",
        font=font(15, True),
        fill=(238, 242, 245),
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    write_preview_provenance(output, manifest, reference)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = render_preview(args.manifest, args.output)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(output)
    print(review_sheet_path(output))
    print(preview_provenance_path(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
