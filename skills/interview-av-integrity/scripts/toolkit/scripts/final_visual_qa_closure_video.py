#!/usr/bin/env python3
"""Independent visual/audio QA for the rendered closure-evidence video.

This script intentionally checks the encoded MP4, not the pre-encode canvases.
It creates:

* one intro/outro contact sheet;
* one six-frame (normal/slow x start/release/end) sheet per case;
* one all-native-frame mouth-ROI timeline per case;
* per-phase audio RMS/dropout measurements;
* a JSON summary consumed by the human QA report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RED = np.array([255, 77, 90], dtype=np.int16)
GREEN = np.array([66, 212, 126], dtype=np.int16)
CYAN = np.array([88, 214, 255], dtype=np.int16)


def font(size: int):
    candidates = [
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"),
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_28 = font(28)
FONT_22 = font(22)
FONT_18 = font(18)


def frame_index(time_s: float, fps: float) -> int:
    return int(round(time_s * fps))


def _color_count(image: Image.Image, target: np.ndarray, tolerance: int = 42) -> int:
    array = np.asarray(image, dtype=np.int16)
    distance = np.max(np.abs(array - target[None, None, :]), axis=2)
    return int(np.count_nonzero(distance <= tolerance))


def _make_labeled_grid(
    cells: list[tuple[str, Image.Image]],
    *,
    columns: int,
    cell_width: int,
    image_height: int,
    heading: str,
) -> Image.Image:
    label_height = 44
    heading_height = 58
    rows = int(math.ceil(len(cells) / columns))
    sheet = Image.new("RGB", (columns * cell_width, heading_height + rows * (image_height + label_height)), "#080c12")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), heading, fill="white", font=FONT_28)
    for index, (label, image) in enumerate(cells):
        col = index % columns
        row = index // columns
        x = col * cell_width
        y = heading_height + row * (image_height + label_height)
        fitted = image.resize((cell_width, image_height), Image.Resampling.LANCZOS)
        sheet.paste(fitted, (x, y))
        draw.rectangle((x, y + image_height, x + cell_width, y + image_height + label_height), fill="#111923")
        draw.text((x + 10, y + image_height + 9), label, fill="white", font=FONT_18)
    return sheet


def _audio_array(frame: av.AudioFrame) -> np.ndarray:
    x = frame.to_ndarray()
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] not in (1, 2) and x.shape[1] in (1, 2):
        x = x.T
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        x = x.astype(np.float32, copy=False)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    return np.ascontiguousarray(x[:2])


def decode_audio(path: Path, duration_s: float, rate: int) -> np.ndarray:
    audio = np.zeros((2, int(math.ceil(duration_s * rate)) + 4096), dtype=np.float32)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            x = _audio_array(frame)
            start = int(round(float(frame.pts * frame.time_base) * rate))
            src_start = max(0, -start)
            dst_start = max(0, start)
            count = min(x.shape[1] - src_start, audio.shape[1] - dst_start)
            if count > 0:
                audio[:, dst_start : dst_start + count] = x[:, src_start : src_start + count]
    return audio


def audio_stats(audio: np.ndarray, start_s: float, end_s: float, rate: int) -> dict:
    start = max(0, int(round(start_s * rate)))
    end = min(audio.shape[1], int(round(end_s * rate)))
    x = audio[:, start:end].astype(np.float64, copy=False)
    mono = np.mean(x, axis=0) if x.size else np.zeros(0)
    block_samples = int(round(0.02 * rate))
    block_rms = []
    for offset in range(0, mono.size, block_samples):
        block = mono[offset : offset + block_samples]
        if block.size:
            block_rms.append(float(np.sqrt(np.mean(block * block))))
    near_zero = [value < 1e-5 for value in block_rms]
    longest = current = 0
    for value in near_zero:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "rms": float(np.sqrt(np.mean(x * x))) if x.size else 0.0,
        "peak": float(np.max(np.abs(x))) if x.size else 0.0,
        "block_ms": 20,
        "near_zero_block_count": int(sum(near_zero)),
        "block_count": len(block_rms),
        "max_consecutive_near_zero_ms": int(longest * 20),
        "nonzero_fraction": float(np.count_nonzero(np.abs(x) > 1e-7) / x.size) if x.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    case_count = len(cases)
    render = manifest["render"]
    fps = float(render["fps"])
    rate = int(render["audio_sample_rate"])
    frame_width = int(render["width"])
    frame_height = int(render["height"])
    slow_repeat = int(render["slow_source_frame_repeat"])
    inset = tuple(int(value) for value in render["layout"]["mouth_inset_xyxy"])
    if len(inset) != 4:
        raise ValueError("manifest mouth inset must contain four coordinates")
    preview_width = 640
    preview_height = max(1, round(preview_width * frame_height / frame_width))
    duration_s = float(manifest["output"]["expected_duration_s"])
    total_frames = int(manifest["output"]["total_frame_count"])

    with args.frame_map.open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    assert len(frame_rows) == total_frames

    targets: dict[int, list[tuple[str, str]]] = {}
    checkpoint_meta: dict[str, list[dict]] = {}
    for case in cases:
        event_id = case["event_id"]
        checkpoint_meta[event_id] = []
        for phase_name in ("normal", "slow"):
            phase = case[phase_name]
            points = {
                "start": frame_index(phase["output_start_s"], fps),
                "release": frame_index(phase["marker_output_s"], fps),
                "end": frame_index(phase["output_end_s"], fps) - 1,
            }
            phase_start = frame_index(phase["output_start_s"], fps)
            phase_end = frame_index(phase["output_end_s"], fps) - 1
            for point_name, index in points.items():
                index = min(phase_end, max(phase_start, index))
                targets.setdefault(index, []).append((event_id, f"{phase_name}_{point_name}"))
                checkpoint_meta[event_id].append({"phase": phase_name, "point": point_name, "frame_index": index})

    intro_frames = int(manifest["cards"]["intro_frames"])
    outro_frames = int(manifest["cards"]["outro_frames"])
    outro_start = total_frames - outro_frames
    card_targets: dict[int, str] = {}
    if intro_frames:
        card_targets.update({
            0: "intro_start",
            intro_frames // 2: "intro_middle",
            intro_frames - 1: "intro_end",
        })
    if outro_frames:
        card_targets.update({
            outro_start: "outro_start",
            outro_start + outro_frames // 2: "outro_middle",
            total_frames - 1: "outro_end",
        })
    for index, name in card_targets.items():
        targets.setdefault(index, []).append(("cards", name))

    checkpoint_images: dict[tuple[str, str], Image.Image] = {}
    roi_images: dict[str, list[tuple[int, float, Image.Image]]] = {case["event_id"]: [] for case in cases}
    encoded_roi_signatures: dict[str, dict[str, list[tuple[int, np.ndarray]]]] = {
        case["event_id"]: {"normal": [], "slow": []} for case in cases
    }
    frame_colors: dict[str, dict] = {}
    phase_ranges = {
        case["event_id"]: (
            frame_index(case["normal"]["output_start_s"], fps),
            frame_index(case["normal"]["output_end_s"], fps) - 1,
        )
        for case in cases
    }

    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        for decoded_index, frame in enumerate(container.decode(stream)):
            if decoded_index >= total_frames:
                raise RuntimeError("decoded more frames than manifest")
            image = frame.to_image().convert("RGB")
            if image.size != (frame_width, frame_height):
                raise RuntimeError(
                    f"decoded dimensions {image.size} differ from manifest "
                    f"{frame_width}x{frame_height}"
                )
            row = frame_rows[decoded_index]
            row_event = row.get("event_id", "")
            row_phase = row.get("phase", "")
            if row_event in encoded_roi_signatures and row_phase in {"normal", "slow"}:
                signature = np.asarray(
                    image.crop(inset).resize((64, 42), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                encoded_roi_signatures[row_event][row_phase].append(
                    (int(row["source_frame_index"]), signature)
                )
            if decoded_index in targets:
                downscaled = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
                for event_id, name in targets[decoded_index]:
                    checkpoint_images[(event_id, name)] = downscaled.copy()
                    key = f"{event_id}:{name}"
                    frame_colors[key] = {
                        "red_pixels": _color_count(image, RED),
                        "green_pixels": _color_count(image, GREEN),
                        "cyan_pixels": _color_count(image, CYAN),
                    }
            for case in cases:
                event_id = case["event_id"]
                first, last = phase_ranges[event_id]
                if first <= decoded_index <= last:
                    crop = image.crop(inset).resize((240, 156), Image.Resampling.LANCZOS)
                    source_s = float(frame_rows[decoded_index]["source_pts_s"])
                    roi_images[event_id].append((decoded_index, source_s, crop))
        if decoded_index + 1 != total_frames:
            raise RuntimeError(f"decoded {decoded_index + 1}, expected {total_frames}")

    card_cells = [(name.replace("_", " "), checkpoint_images[("cards", name)]) for name in card_targets.values()]
    generated: list[str] = []
    if card_cells:
        card_sheet = _make_labeled_grid(
            card_cells,
            columns=3,
            cell_width=preview_width,
            image_height=preview_height,
            heading="Intro / outro encoded-frame checkpoints",
        )
        card_sheet.save(args.output_dir / "FINAL_VISUAL_QA_intro_outro.png")
        generated.append("FINAL_VISUAL_QA_intro_outro.png")
    for case in cases:
        event_id = case["event_id"]
        cells = []
        for phase_name in ("normal", "slow"):
            for point_name in ("start", "release", "end"):
                key = f"{phase_name}_{point_name}"
                cells.append((key.replace("_", " "), checkpoint_images[(event_id, key)]))
        filename = f"FINAL_VISUAL_QA_case_{case['order']:02d}_{event_id}.png"
        sheet = _make_labeled_grid(
            cells,
            columns=3,
            cell_width=preview_width,
            image_height=preview_height,
            heading=f"Case {case['order']}/{case_count} | {event_id} | {case['presentation_label']}",
        )
        sheet.save(args.output_dir / filename)
        generated.append(filename)

        roi_cells = []
        for relative, (index, source_s, crop) in enumerate(roi_images[event_id]):
            roi_cells.append((f"f{relative:02d}  src {source_s:.3f}s", crop))
        roi_filename = f"FINAL_VISUAL_QA_roi_all_frames_{case['order']:02d}_{event_id}.png"
        roi_sheet = _make_labeled_grid(
            roi_cells,
            columns=8,
            cell_width=240,
            image_height=156,
            heading=(
                f"All {len(roi_cells)} normal frames; "
                f"slow repeats each x{slow_repeat} | {event_id}"
            ),
        )
        roi_sheet.save(args.output_dir / roi_filename)
        generated.append(roi_filename)

    audio = decode_audio(args.video, duration_s + 0.1, rate)
    audio_rows = []
    for case in cases:
        for phase_name in ("normal", "slow"):
            phase = case[phase_name]
            stats = audio_stats(
                audio,
                float(phase["output_start_s"]),
                float(phase["output_end_s"]),
                rate,
            )
            audio_rows.append(
                {
                    "case_order": case["order"],
                    "event_id": case["event_id"],
                    "phase": phase_name,
                    "start_s": phase["output_start_s"],
                    "end_s": phase["output_end_s"],
                    **stats,
                }
            )
    audio_csv = args.output_dir / "FINAL_VISUAL_QA_audio_phase_rms.csv"
    with audio_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audio_rows[0]))
        writer.writeheader()
        writer.writerows(audio_rows)
    generated.append(audio_csv.name)

    slow_roi_reuse = {}
    for case in cases:
        event_id = case["event_id"]
        normal = encoded_roi_signatures[event_id]["normal"]
        slow = encoded_roi_signatures[event_id]["slow"]
        normal_by_source = {source_index: signature for source_index, signature in normal}
        counts: dict[int, int] = {}
        differences = []
        unmatched = []
        for source_index, signature in slow:
            counts[source_index] = counts.get(source_index, 0) + 1
            reference = normal_by_source.get(source_index)
            if reference is None:
                unmatched.append(source_index)
                continue
            differences.append(float(np.mean(np.abs(signature.astype(np.int16) - reference.astype(np.int16)))))
        slow_roi_reuse[event_id] = {
            "normal_frame_count": len(normal),
            "slow_frame_count": len(slow),
            "normal_unique_source_frames": len(normal_by_source),
            "slow_unique_source_frames": len(counts),
            "all_slow_sources_matched_normal": not unmatched,
            "all_slow_sources_repeated_four_times": bool(counts) and set(counts.values()) == {4},
            "encoded_inset_mean_abs_difference_mean": float(np.mean(differences)) if differences else None,
            "encoded_inset_mean_abs_difference_p95": float(np.percentile(differences, 95)) if differences else None,
            "encoded_inset_mean_abs_difference_max": float(np.max(differences)) if differences else None,
        }

    summary = {
        "video": str(args.video),
        "manifest": str(args.manifest),
        "frame_map": str(args.frame_map),
        "decoded_frames": total_frames,
        "render": {
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "audio_sample_rate": rate,
            "mouth_inset_xyxy": list(inset),
            "slow_source_frame_repeat": slow_repeat,
        },
        "cases": [
            {
                "order": case["order"],
                "event_id": case["event_id"],
                "classification": case["classification"],
                "normal_unique_roi_frames": len(roi_images[case["event_id"]]),
                "checkpoints": checkpoint_meta[case["event_id"]],
            }
            for case in cases
        ],
        "audio_phase_stats": audio_rows,
        "slow_roi_reuse": slow_roi_reuse,
        "encoded_frame_color_counts": frame_colors,
        "generated_files": generated,
    }
    (args.output_dir / "FINAL_VISUAL_QA_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
