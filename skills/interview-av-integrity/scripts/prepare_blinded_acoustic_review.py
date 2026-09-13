#!/usr/bin/env python3
"""Create audio-only, speaker-blinded plosive review sheets and a blank CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TOOLKIT = Path(__file__).resolve().parent / "toolkit"
sys.path.insert(0, str(TOOLKIT))
from video_integrity_analyzer.plosive_sync import decode_audio_window  # noqa: E402


def nested(value: object, key: str) -> dict:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return value[key]
    return {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False):
    paths = [
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc" if bold else "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def save_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    values = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    peak = max(float(np.max(np.abs(values))), 1e-9)
    pcm = np.clip(values / max(peak, 1.0), -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def covered_audio_intervals(
    waveform: np.ndarray, coverage_mask: np.ndarray, start_s: float, sample_rate: int
) -> list[dict[str, float]]:
    """Keep decoded finite sample runs; timeline padding is not observed audio."""
    values = np.asarray(waveform).reshape(-1)
    covered = np.asarray(coverage_mask, dtype=bool).reshape(-1)
    if covered.size != values.size:
        raise ValueError("coverage_mask must match waveform")
    covered = covered & np.isfinite(values)
    edges = np.diff(np.pad(covered.astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        {"start_s": start_s + int(start) / sample_rate, "end_s": start_s + int(end) / sample_rate}
        for start, end in zip(starts, ends)
    ]


def time_is_covered(time_s: float, intervals: list[dict[str, float]]) -> bool:
    # Stabilize half-open boundaries against floating-point anchor arithmetic.
    time_s = round(time_s, 9)
    return any(round(interval["start_s"], 9) <= time_s < round(interval["end_s"], 9) for interval in intervals)


def spectrogram(values: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0)
    frame = max(128, int(round(sample_rate * 0.008)))
    fft_size = 1024
    hop = max(1, int(round(sample_rate * 0.002)))
    if len(values) < frame:
        values = np.pad(values, (0, frame - len(values)))
    starts = range(0, len(values) - frame + 1, hop)
    window = np.hanning(frame)
    spectra = [np.abs(np.fft.rfft(values[start : start + frame] * window, n=fft_size)) ** 2 for start in starts]
    matrix = np.stack(spectra, axis=1) if spectra else np.zeros((fft_size // 2 + 1, 1))
    top = min(matrix.shape[0], int(round(12_000 / (sample_rate / fft_size))) + 1)
    db = 10.0 * np.log10(np.maximum(matrix[:top], 1e-12))
    low, high = np.percentile(db, [5, 99.5])
    normalized = np.clip((db - low) / max(high - low, 1e-9), 0.0, 1.0)
    return normalized[::-1]


def colorize(gray: np.ndarray) -> Image.Image:
    x = np.clip(gray, 0.0, 1.0)
    red = np.clip(1.8 * x - 0.55, 0, 1)
    green = np.clip(1.7 * x, 0, 1)
    blue = np.clip(0.25 + 1.4 * (1 - np.abs(2 * x - 1)), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def render_sheet(
    blind_id: str,
    waveform: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    anchor_s: float,
    candidates: list[dict],
    phone: str,
    kana: str,
    word: str,
    covered_intervals: list[dict[str, float]] | None = None,
) -> Image.Image:
    width, height = 1400, 820
    left, right = 92, 1368
    wave_top, wave_bottom = 112, 310
    spec_top, spec_bottom = 380, 742
    image = Image.new("RGB", (width, height), "#0a1017")
    draw = ImageDraw.Draw(image)
    draw.text((32, 22), f"{blind_id}  target /{phone}/ {kana}  {word}", fill="white", font=font(30, True))
    draw.text((32, 66), "audio only | speaker/group/video hidden | times relative to ASR anchor", fill="#a8b3c2", font=font(20))

    x_values = np.linspace(left, right, max(len(waveform), 2))
    clean = np.nan_to_num(waveform, nan=0.0)
    scale = np.percentile(np.abs(clean), 99.5) if clean.size else 1.0
    clean = np.clip(clean / max(float(scale), 1e-9), -1.0, 1.0)
    center = (wave_top + wave_bottom) / 2
    y_values = center - clean * (wave_bottom - wave_top) * 0.45
    points = [(int(x), int(y)) for x, y in zip(x_values[::max(1, len(x_values)//4000)], y_values[::max(1, len(y_values)//4000)])]
    draw.rectangle((left, wave_top, right, wave_bottom), outline="#334255", width=2)
    if len(points) > 1:
        draw.line(points, fill="#58d6ff", width=2)

    spec = colorize(spectrogram(waveform, sample_rate)).resize((right - left, spec_bottom - spec_top), Image.Resampling.BILINEAR)
    image.paste(spec, (left, spec_top))
    draw.rectangle((left, spec_top, right, spec_bottom), outline="#334255", width=2)

    duration = end_s - start_s
    def xpos(time_s: float) -> int:
        return int(round(left + (time_s - start_s) / duration * (right - left)))

    if covered_intervals is not None:
        cursor = start_s
        gaps = []
        for interval in covered_intervals:
            if cursor < interval["start_s"]:
                gaps.append((cursor, interval["start_s"]))
            cursor = interval["end_s"]
        if cursor < end_s:
            gaps.append((cursor, end_s))
        for gap_start, gap_end in gaps:
            for top, bottom in ((wave_top, wave_bottom), (spec_top, spec_bottom)):
                draw.rectangle((xpos(gap_start), top, xpos(gap_end), bottom), fill="#392a30")
        if gaps:
            draw.text((left, wave_bottom + 16), "Shaded intervals: no decoded audio; do not annotate a release there.", fill="#ffb4bc", font=font(18))

    anchor_x = xpos(anchor_s)
    draw.line((anchor_x, wave_top, anchor_x, spec_bottom), fill="#ffffff", width=2)
    draw.text((anchor_x + 5, spec_bottom + 9), "anchor 0 ms", fill="white", font=font(18))
    colors = ("#ffcc00", "#ff6b6b", "#7ee787")
    for rank, candidate in enumerate(candidates[:3], start=1):
        time_s = float(candidate["time_s"])
        x = xpos(time_s)
        color = colors[rank - 1]
        draw.line((x, wave_top, x, spec_bottom), fill=color, width=3)
        relative = (time_s - anchor_s) * 1000.0
        draw.text((x + 4, wave_top + 8 + (rank - 1) * 25), f"C{rank} {relative:+.0f} ms", fill=color, font=font(17, True))
    for offset_ms in range(-400, 321, 100):
        time_s = anchor_s + offset_ms / 1000.0
        if start_s <= time_s <= end_s:
            x = xpos(time_s)
            draw.line((x, spec_bottom, x, spec_bottom + 7), fill="#8b98a8", width=1)
            draw.text((x - 22, spec_bottom + 31), str(offset_ms), fill="#8b98a8", font=font(15))
    draw.text((left, height - 28), "Review: choose C-rank or enter a relative release time; use unmeasurable when the target release is not defensible.", fill="#a8b3c2", font=font(17))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--window-before", type=float, default=0.45)
    parser.add_argument("--window-after", type=float, default=0.35)
    parser.add_argument("--hide-word", action="store_true")
    parser.add_argument("--max-events", type=int, help="deterministic smoke-test limit after shuffling")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    events_path = args.events.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not video.is_file() or not events_path.is_file():
        raise SystemExit("video and events JSON must exist")
    if not all(math.isfinite(value) and value > 0 for value in (args.window_before, args.window_after)):
        raise SystemExit("review windows must be positive")

    payload = json.loads(events_path.read_text(encoding="utf-8"))
    source_events = payload.get("events")
    if not isinstance(source_events, list):
        raise SystemExit("events JSON must contain an events list")

    reviewable = []
    skipped = []
    for event in source_events:
        selection = nested(event, "selection")
        acoustic = nested(nested(event, "measurement"), "acoustic")
        candidates = acoustic.get("candidates")
        event_id = str(selection.get("event_id") or "")
        if selection.get("eligible") is True and isinstance(candidates, list):
            reviewable.append(event)
        else:
            skipped.append({"event_id": event_id, "reason": "not eligible or acoustic analysis unavailable"})
    if not reviewable:
        raise SystemExit("no reviewable acoustic events")
    reviewable.sort(key=lambda event: str(nested(event, "selection").get("event_id")))
    random.Random(args.seed).shuffle(reviewable)
    reviewable_total = len(reviewable)
    deferred_by_limit: list[str] = []
    if args.max_events is not None:
        if args.max_events <= 0:
            raise SystemExit("--max-events must be positive")
        deferred_by_limit = [
            str(nested(event, "selection").get("event_id"))
            for event in reviewable[args.max_events :]
        ]
        reviewable = reviewable[: args.max_events]
    output_dir.mkdir(parents=True, exist_ok=True)

    key_rows = []
    csv_rows = []
    decode_failures = []
    for index, event in enumerate(reviewable, start=1):
        blind_id = f"AR-{index:04d}"
        selection = nested(event, "selection")
        acoustic = nested(nested(event, "measurement"), "acoustic")
        anchor = float(selection["anchor_s"])
        start_s = max(0.0, anchor - args.window_before)
        end_s = anchor + args.window_after
        try:
            audio = decode_audio_window(video, start_s=start_s, end_s=end_s, sample_rate=48_000)
            covered_intervals = covered_audio_intervals(
                audio.waveform, audio.coverage_mask, audio.start_s, audio.sample_rate
            )
            candidates = [
                candidate for candidate in acoustic["candidates"]
                if isinstance(candidate, dict)
                and audio.start_s <= float(candidate["time_s"]) < audio.end_s
                and time_is_covered(float(candidate["time_s"]), covered_intervals)
            ]
            word = "" if args.hide_word else str(selection.get("token_text") or "")
            sheet = render_sheet(
                blind_id,
                audio.waveform,
                audio.sample_rate,
                audio.start_s,
                audio.end_s,
                anchor,
                candidates,
                str(selection.get("phoneme_class") or ""),
                str(selection.get("kana") or ""),
                word,
                covered_intervals=covered_intervals,
            )
            sheet.save(output_dir / f"{blind_id}.png")
            save_wav(output_dir / f"{blind_id}.wav", audio.waveform, audio.sample_rate)
        except Exception as error:
            decode_failures.append({"blind_id": blind_id, "event_id": selection.get("event_id"), "error": f"{type(error).__name__}: {error}"})
            continue
        candidate_rows = [
            {
                "rank": rank,
                "time_s": float(candidate["time_s"]),
                "relative_ms": (float(candidate["time_s"]) - anchor) * 1000.0,
                "score": candidate.get("score"),
                "energy_rise_db": candidate.get("energy_rise_db"),
                "energy_slope_12ms_db": candidate.get("energy_slope_12ms_db"),
                "hf_rise_db": candidate.get("high_frequency_rise_db"),
            }
            for rank, candidate in enumerate(candidates[:3], start=1)
        ]
        key_rows.append(
            {
                "blind_id": blind_id,
                "runner_event_id": selection["event_id"],
                "anchor_s": anchor,
                "review_window": {
                    "start_s": audio.start_s,
                    "end_s": audio.end_s,
                    "covered_intervals": covered_intervals,
                },
                "speaker": selection.get("speaker"),
                "group": selection.get("group"),
                "epoch_id": selection.get("epoch_id"),
                "phoneme_class": selection.get("phoneme_class"),
                "target_kana": selection.get("kana"),
                "target_word": selection.get("token_text"),
                "candidates": candidate_rows,
            }
        )
        csv_rows.append(
            {
                "blind_id": blind_id,
                "target_phone": selection.get("phoneme_class"),
                "target_kana": selection.get("kana"),
                "target_word": "" if args.hide_word else selection.get("token_text"),
                "status": "",
                "selected_candidate_rank": "",
                "selected_release_relative_ms": "",
                "acoustic_realization": "",
                "confidence": "",
                "reason": "",
            }
        )

    if not key_rows:
        raise SystemExit("all review clips failed to decode")
    key_path = output_dir / "blind_key.json"
    key_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "seed": args.seed,
                "blinding": "video, speaker and candidate/control group absent from review sheets",
                "source_events_sha256": sha256_file(events_path),
                "source_video_sha256": sha256_file(video),
                "reviewable_total": reviewable_total,
                "selected_count": len(key_rows),
                "max_events": args.max_events,
                "deferred_by_max_events": deferred_by_limit,
                "events": key_rows,
                "skipped": skipped,
                "decode_failures": decode_failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "acoustic_annotations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Prepared {len(key_rows)} blinded events; skipped {len(skipped)}; decode failures {len(decode_failures)}")
    print(key_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
