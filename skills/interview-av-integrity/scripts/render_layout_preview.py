#!/usr/bin/env python3
"""Render a local synthetic A/V layout preview through the production frame path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
TOOLKIT_DIR = SCRIPT_DIR / "toolkit"
TOOLKIT_SCRIPTS = TOOLKIT_DIR / "scripts"
for directory in (TOOLKIT_SCRIPTS, TOOLKIT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import render_closure_evidence_video as renderer  # noqa: E402
from video_integrity_analyzer.layout_reference import (  # noqa: E402
    REQUIRED_QUALITY_CHECKS,
    load_approved_layout_reference,
    render_manifest_basis_sha256,
    renderer_source_records,
    sha256_file,
)


REVIEW_CHECKS = [
    "top row preserves case count, finding, speed, and red/green legend hierarchy",
    "source panel remains dominant and the same-frame mouth inset is easy to trace",
    "mouth ROI and inset are large enough to judge visible lip contact",
    "waveform panel clearly separates the closure-review window, burst, release marker, and playhead",
    "Japanese labels and limitation text remain readable at 1920x1080",
    "differences are limited to dummy content or documented case-dependent adaptations",
]


def _synthetic_source_frame() -> Image.Image:
    """Create a plainly synthetic source placeholder; no case media is read."""

    width, height = renderer.OUTPUT_WIDTH, renderer.OUTPUT_HEIGHT
    image = Image.new("RGB", (width, height), "#b7afa2")
    draw = ImageDraw.Draw(image)

    # Abstract room blocks.
    draw.rectangle((0, 0, width, 680), fill="#c9c1b4")
    draw.rectangle((0, 680, width, height), fill="#786b60")
    draw.rectangle((80, 150, 430, 650), fill="#9d8e7f")
    draw.rounded_rectangle((120, 210, 385, 625), radius=24, fill="#d8cbbb")
    draw.rectangle((1510, 95, 1840, 690), fill="#6d6259")
    for row in range(4):
        draw.rectangle(
            (1540, 135 + row * 125, 1810, 215 + row * 125),
            fill=(82 + row * 10, 73 + row * 8, 67 + row * 7),
        )
    draw.ellipse((300, 230, 510, 520), fill="#6f875d")
    draw.rectangle((386, 465, 425, 675), fill="#7e6753")

    # Abstract candidate; deliberately illustrative rather than photographic.
    draw.rounded_rectangle((630, 605, 1290, 1080), radius=160, fill="#17283e")
    draw.ellipse((760, 80, 1160, 650), fill="#d2a78d")
    draw.pieslice((740, 40, 1180, 410), 180, 360, fill="#28313b")
    draw.ellipse((840, 300, 880, 325), fill="#29303a")
    draw.ellipse((1040, 300, 1080, 325), fill="#29303a")
    draw.line((955, 320, 940, 410, 978, 418), fill="#9a735f", width=8)
    draw.ellipse((900, 474, 1020, 532), fill="#6e3540")
    draw.ellipse((915, 488, 1005, 520), fill="#1f1517")

    font = renderer.make_fonts().small
    draw.rounded_rectangle((64, 246, 625, 310), radius=14, fill="#0b1119")
    draw.text(
        (88, 264),
        "SYNTHETIC PLACEHOLDER — 実案件媒体なし",
        fill="#f7f9fc",
        font=font,
    )
    return image


def _synthetic_envelope(width: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, endpoint=False)
    baseline = 0.045 + 0.018 * np.sin(2.0 * math.pi * 11.0 * x) ** 2
    lead = 0.16 * np.exp(-((x + 0.20) / 0.08) ** 2)
    burst = 0.92 * np.exp(-((x - 0.01) / 0.028) ** 2)
    tail = 0.28 * np.exp(-((x - 0.17) / 0.19) ** 2)
    return np.clip(baseline + lead + burst + tail, 0.0, 1.0).astype(np.float32)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def _review_sheet(reference: Image.Image, preview: Image.Image) -> Image.Image:
    width, height = 1920, 660
    canvas = Image.new("RGB", (width, height), "#080c12")
    draw = ImageDraw.Draw(canvas)
    fonts = renderer.make_fonts()
    draw.text(
        (width // 2, 22),
        "A/V LAYOUT QUALITY GATE — 参照と本番renderer previewを比較",
        anchor="ma",
        fill="#f7f9fc",
        font=fonts.body,
    )
    left_bounds = (28, 98, 946, 615)
    right_bounds = (974, 98, 1892, 615)
    for bounds, source, label, color in (
        (left_bounds, reference, "APPROVED REFERENCE", "#f4c95d"),
        (right_bounds, preview, "PRODUCTION-PATH PREVIEW", "#58d6ff"),
    ):
        x0, y0, x1, y1 = bounds
        fitted = _fit(source, (x1 - x0, y1 - y0))
        x = x0 + (x1 - x0 - fitted.width) // 2
        y = y0 + (y1 - y0 - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.rounded_rectangle((x0, 62, x0 + 330, 94), radius=8, fill="#111a25")
        draw.text((x0 + 14, 69), label, fill=color, font=fonts.tiny)
    draw.text(
        (width // 2, 635),
        "ダミー文言ではなく、情報階層・領域比率・口元導線・波形強調・可読性を確認",
        anchor="mm",
        fill="#a8b3c2",
        font=fonts.tiny,
    )
    return canvas


def _write_image(path: Path, image: Image.Image, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing preview artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def render_layout_preview(
    manifest: Path,
    output: Path,
    *,
    review_sheet: Path | None = None,
    report: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render implementation preview, side-by-side sheet, and pending gate report."""

    manifest = manifest.expanduser().resolve()
    renderer.load_manifest(manifest)
    reference = load_approved_layout_reference(SKILL_ROOT)
    reference_path = SKILL_ROOT / str(reference["relative_path"])
    output = output.expanduser().resolve()
    review_sheet = (
        review_sheet.expanduser().resolve()
        if review_sheet is not None
        else output.with_name(f"{output.stem}_review_sheet.png")
    )
    report = (
        report.expanduser().resolve()
        if report is not None
        else output.with_name(f"{output.stem}_quality_gate.json")
    )
    if len({output, review_sheet, report}) != 3:
        raise ValueError("preview, review sheet, and report paths must be distinct")
    if output.suffix.lower() != ".png" or review_sheet.suffix.lower() != ".png":
        raise ValueError("preview and review-sheet outputs must use .png")
    if report.suffix.lower() != ".json":
        raise ValueError("preview provenance output must use .json")
    existing = [path for path in (output, review_sheet, report) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing preview artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    event = renderer.EvidenceEvent(
        event_id="synthetic-layout-preview",
        release_s=12.0,
        category="missing",
        label="合成サンプル /p/",
        note="本番rendererの情報階層と口元拡大を確認",
        pre_s=19 / renderer.OUTPUT_FPS,
        post_s=19 / renderer.OUTPUT_FPS,
        mouth_roi=(0.43, 0.36, 0.57, 0.58),
        order=1,
    )
    source = renderer.SourceFrame(
        time_s=event.release_s,
        source_frame_index=int(event.release_s * renderer.OUTPUT_FPS),
        image=_synthetic_source_frame(),
    )
    preview = renderer.render_evidence_frame(
        source,
        event,
        _synthetic_envelope(renderer.OUTPUT_WIDTH - 144),
        "slow",
        3,
        7,
        3,
        5,
        renderer.make_fonts(),
        "/p/",
    )
    _write_image(output, preview, overwrite=overwrite)
    with Image.open(reference_path) as image:
        reference_image = image.convert("RGB")
    sheet = _review_sheet(reference_image, preview)
    _write_image(review_sheet, sheet, overwrite=overwrite)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "av_integrity_production_renderer_layout_preview",
        "review_status": "PENDING_USER_APPROVAL",
        "scope": "layout structure and visual quality only; not case evidence",
        "case_media_used": False,
        "new_imagegen_concept_used": False,
        "render_manifest_basis_sha256": render_manifest_basis_sha256(manifest),
        "production_render_path": (
            "scripts/toolkit/scripts/render_closure_evidence_video.py::"
            "render_evidence_frame"
        ),
        "layout_reference": reference,
        "renderer_sources": renderer_source_records(SKILL_ROOT),
        "implementation_preview": {
            "path": str(output),
            "sha256": sha256_file(output),
            "width": preview.width,
            "height": preview.height,
        },
        "side_by_side_review_sheet": {
            "path": str(review_sheet),
            "sha256": sha256_file(review_sheet),
            "width": sheet.width,
            "height": sheet.height,
        },
        "quality_contract": {
            "reference_is_structure_and_visual_quality_only": True,
            "reference_text_people_and_values_are_non_authoritative": True,
            "production_canvas": {"width": 1920, "height": 1080},
            "required_human_checks": list(REQUIRED_QUALITY_CHECKS),
            "human_review_guidance": REVIEW_CHECKS,
        },
        "next_action": (
            "Show the approved reference and production-path preview together. "
            "Do not start the full render until the user explicitly approves the preview."
        ),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-sheet", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render_layout_preview(
            args.manifest,
            args.output,
            review_sheet=args.review_sheet,
            report=args.report,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Preview: {result['implementation_preview']['path']}")
    print(f"Review sheet: {result['side_by_side_review_sheet']['path']}")
    print("Gate: PENDING_USER_APPROVAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
