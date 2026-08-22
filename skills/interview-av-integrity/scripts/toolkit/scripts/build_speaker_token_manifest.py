#!/usr/bin/env python3
"""Build active-speaker intervals and a bilabial-token manifest.

The recording is a speaker-view export.  This script identifies the displayed
participant from the *name label only*: it crops a configured label region,
suppresses its slowly varying background, and compares the remaining letter
strokes with user-supplied reference label crops.  It deliberately does not use
face appearance, voice characteristics, accent, or any other personal trait.

The ASR transcript is used only to pre-select Japanese /p/ and /b/ kana. For a
complete inventory, every word must include a normalized ``reading`` (or
``kana``/``pronunciation``) field. Word timestamps are not treated as acoustic
release times; downstream code must refine them from the waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


BILABIAL_KANA_RE = re.compile(r"[パピプペポぱぴぷぺぽバビブベボばびぶべぼ]")
ANALYSIS_GROUPS = frozenset({"candidate", "control"})
ALLOWED_GROUPS = frozenset((*ANALYSIS_GROUPS, "exclude"))


@dataclass(frozen=True)
class CropGeometry:
    x0: int = 10
    y0: int = 1000
    x1: int = 200
    y1: int = 1070

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_references(path: Path) -> dict[str, dict[str, object]]:
    """Load and validate the auditable speaker-label reference configuration."""

    resolved = path.expanduser().resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) < 2:
        raise ValueError(
            "references JSON must map at least two speaker IDs to reference objects"
        )
    references: dict[str, dict[str, object]] = {}
    for raw_speaker_id, raw_reference in data.items():
        speaker_id = str(raw_speaker_id).strip()
        if not speaker_id or speaker_id == "unknown":
            raise ValueError("reference speaker IDs must be non-empty and not 'unknown'")
        if not isinstance(raw_reference, dict):
            raise ValueError(f"reference {speaker_id!r} must be an object")
        display_name = str(raw_reference.get("display_name", "")).strip()
        group = str(raw_reference.get("group", "")).strip()
        try:
            time_s = float(raw_reference["time_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"reference {speaker_id!r} must contain a finite time_s"
            ) from error
        if not display_name:
            raise ValueError(f"reference {speaker_id!r} has an empty display_name")
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError(f"reference {speaker_id!r} has invalid time_s")
        if group not in ALLOWED_GROUPS:
            allowed = ", ".join(sorted(ALLOWED_GROUPS))
            raise ValueError(
                f"reference {speaker_id!r} group must be one of: {allowed}"
            )
        references[speaker_id] = {
            "display_name": display_name,
            "time_s": time_s,
            "group": group,
        }
    configured_groups = {str(value["group"]) for value in references.values()}
    missing_groups = ANALYSIS_GROUPS - configured_groups
    if missing_groups:
        raise ValueError(
            "references JSON must contain at least one speaker in each analysis "
            f"group; missing: {', '.join(sorted(missing_groups))}"
        )
    return references


def _media_duration_s(video_path: Path) -> float:
    with av.open(str(video_path)) as container:
        if container.duration is not None:
            duration_s = float(container.duration / av.time_base)
        else:
            stream_durations = [
                float(stream.duration * stream.time_base)
                for stream in container.streams
                if stream.duration is not None and stream.time_base is not None
            ]
            duration_s = max(stream_durations, default=0.0)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("media duration is unavailable")
    return duration_s


def _resolve_analysis_end(
    requested_end_s: float | None, *, media_duration_s: float
) -> float:
    end_s = media_duration_s if requested_end_s is None else float(requested_end_s)
    if not math.isfinite(end_s) or end_s <= 0.0:
        raise ValueError("analysis end must be a positive finite time")
    if end_s > media_duration_s + 1e-6:
        raise ValueError(
            f"analysis end {end_s:.6f}s exceeds media duration "
            f"{media_duration_s:.6f}s"
        )
    return min(end_s, media_duration_s)


def _label_feature(rgb_crop: np.ndarray) -> np.ndarray:
    """Return a binary map of bright label strokes after background removal."""
    gray_image = Image.fromarray(rgb_crop, mode="RGB").convert("L")
    gray = np.asarray(gray_image, dtype=np.float32)
    blurred = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(3.0)), dtype=np.float32)
    positive_high_pass = np.maximum(gray - blurred, 0.0)
    return positive_high_pass > 8.0


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    numerator = 2.0 * float(np.logical_and(left, right).sum())
    denominator = float(left.sum() + right.sum())
    return numerator / denominator if denominator else 0.0


def classify_label(
    rgb_crop: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    minimum_score: float,
    minimum_margin: float,
) -> dict[str, object]:
    feature = _label_feature(rgb_crop)
    scores = {speaker: _dice(feature, template) for speaker, template in templates.items()}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_score = ranked[0]
    runner_up_score = ranked[1][1]
    margin = best_score - runner_up_score
    accepted = best_score >= minimum_score and margin >= minimum_margin
    return {
        "speaker_id": best_speaker if accepted else "unknown",
        "best_template": best_speaker,
        "best_score": round(best_score, 6),
        "margin": round(margin, 6),
        "scores": {key: round(value, 6) for key, value in scores.items()},
    }


def _load_tokens(
    words_json: Path,
    *,
    start_s: float,
    end_s: float,
    require_complete_readings: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    transcript = json.loads(words_json.read_text(encoding="utf-8"))
    tokens: list[dict[str, object]] = []
    token_index = 0
    lexical_word_count = 0
    supplied_reading_count = 0
    missing_reading_examples: list[str] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            word_start = float(word["start"])
            word_end = float(word["end"])
            if word_end < start_s or word_start >= end_s:
                continue
            surface = str(word.get("word", ""))
            if not surface.strip():
                continue
            lexical_word_count += 1
            reading_source = next(
                (
                    key
                    for key in ("reading", "kana", "pronunciation")
                    if str(word.get(key, "")).strip()
                ),
                None,
            )
            if reading_source is not None:
                supplied_reading_count += 1
                reading = str(word[reading_source])
            else:
                reading = surface
                if len(missing_reading_examples) < 20:
                    missing_reading_examples.append(surface)
            matches = list(BILABIAL_KANA_RE.finditer(reading))
            for occurrence_index, match in enumerate(matches):
                # Current transcript items contain one matched kana each.  The
                # interpolation keeps the manifest deterministic if a future
                # ASR item contains more than one.
                fraction = (match.start() + 0.5) / max(len(reading), 1)
                rough_time = word_start + fraction * max(word_end - word_start, 0.0)
                tokens.append(
                    {
                        "token_id": f"pb_{token_index:04d}",
                        "kana": match.group(0),
                        "word": surface,
                        "reading": reading,
                        "reading_source": reading_source or "surface_word",
                        "occurrence_in_word": occurrence_index,
                        "asr_word_start_s": word_start,
                        "asr_word_end_s": word_end,
                        "asr_rough_time_s": rough_time,
                        "asr_probability": float(word.get("probability", math.nan)),
                        "segment_id": int(segment.get("id", -1)),
                        "segment_text": str(segment.get("text", "")),
                    }
                )
                token_index += 1
    missing_reading_count = lexical_word_count - supplied_reading_count
    if require_complete_readings and missing_reading_count:
        examples = ", ".join(repr(value) for value in missing_reading_examples[:5])
        raise ValueError(
            "complete Japanese bilabial inventory requires a reading/kana/"
            f"pronunciation field for every word; missing {missing_reading_count}/"
            f"{lexical_word_count}, examples: {examples}"
        )
    audit = {
        "lexical_word_count": lexical_word_count,
        "supplied_reading_count": supplied_reading_count,
        "missing_reading_count": missing_reading_count,
        "missing_reading_examples": missing_reading_examples,
        "token_inventory_scope": (
            "reading_complete" if missing_reading_count == 0 else "literal_kana_or_supplied_reading_only"
        ),
    }
    return tokens, audit


def _target_times(
    *,
    start_s: float,
    end_s: float,
    sample_hz: float,
    reference_times: Iterable[float],
    tokens: Sequence[dict[str, object]],
    vote_offsets_s: Sequence[float],
    guard_offsets_s: Sequence[float],
) -> list[float]:
    step = 1.0 / sample_hz
    timeline = np.arange(start_s, end_s, step, dtype=np.float64).tolist()
    event_times: list[float] = []
    for token in tokens:
        center = float(token["asr_rough_time_s"])
        for offset in vote_offsets_s:
            time_s = center + offset
            if start_s <= time_s < end_s:
                event_times.append(time_s)
        for offset in guard_offsets_s:
            time_s = center + offset
            if start_s <= time_s < end_s:
                event_times.append(time_s)
    all_times = timeline + list(reference_times) + event_times
    return sorted({round(float(value), 6) for value in all_times})


def _decode_label_crops(
    video_path: Path,
    target_times: Sequence[float],
    geometry: CropGeometry,
) -> tuple[dict[float, np.ndarray], dict[str, object]]:
    if not target_times:
        return {}, {}
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    targets = list(target_times)
    target_index = 0
    crops: dict[float, np.ndarray] = {}
    observed_frame_times: dict[float, float] = {}
    width = int(stream.codec_context.width)
    height = int(stream.codec_context.height)
    if geometry.x1 > width or geometry.y1 > height:
        raise ValueError(
            f"label crop {geometry.box!r} lies outside {width}x{height} video; "
            "supply coordinates for this export"
        )

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * stream.time_base)
        while target_index < len(targets) and frame_time + 1e-9 >= targets[target_index]:
            target = targets[target_index]
            rgb = frame.to_ndarray(format="rgb24")
            crops[target] = rgb[geometry.y0 : geometry.y1, geometry.x0 : geometry.x1].copy()
            observed_frame_times[target] = frame_time
            target_index += 1
        if target_index >= len(targets):
            break
    container.close()
    if target_index < len(targets):
        missing = targets[target_index:]
        raise RuntimeError(f"video ended before {len(missing)} requested label samples")
    metadata = {
        "width": width,
        "height": height,
        "average_rate": float(stream.average_rate) if stream.average_rate else None,
        "max_target_to_frame_delay_ms": round(
            max((observed_frame_times[t] - t) * 1000.0 for t in targets), 3
        ),
    }
    return crops, metadata


def _majority_vote(observations: list[dict[str, object]]) -> tuple[str, bool]:
    accepted = [str(item["speaker_id"]) for item in observations if item["speaker_id"] != "unknown"]
    if not accepted:
        return "unknown", False
    counts = {speaker: accepted.count(speaker) for speaker in set(accepted)}
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    unique_winner = len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return (ranked[0][0] if unique_winner else "unknown"), unique_winner


def _unanimous_speaker(observations: Sequence[dict[str, object]]) -> str:
    speakers = [str(item["speaker_id"]) for item in observations]
    if speakers and "unknown" not in speakers and len(set(speakers)) == 1:
        return speakers[0]
    return "unknown"


def _compress_intervals(
    samples: Sequence[dict[str, object]],
    *,
    start_s: float,
    end_s: float,
    sample_hz: float,
    speaker_groups: dict[str, str],
) -> list[dict[str, object]]:
    if not samples:
        return []
    half_step = 0.5 / sample_hz
    intervals: list[dict[str, object]] = []
    run_start = 0
    for index in range(1, len(samples) + 1):
        at_end = index == len(samples)
        changed = not at_end and samples[index]["speaker_id"] != samples[run_start]["speaker_id"]
        if not at_end and not changed:
            continue
        run = samples[run_start:index]
        interval_start = start_s if run_start == 0 else float(run[0]["time_s"]) - half_step
        interval_end = end_s if at_end else float(samples[index]["time_s"]) - half_step
        intervals.append(
            {
                "interval_id": f"speaker_{len(intervals):04d}",
                "epoch_id": f"epoch_{len(intervals):04d}",
                "start_s": round(max(start_s, interval_start), 3),
                "end_s": round(min(end_s, interval_end), 3),
                "speaker_id": run[0]["speaker_id"],
                "speaker": run[0]["speaker_id"],
                "group": speaker_groups.get(str(run[0]["speaker_id"]), "exclude"),
                "sample_count": len(run),
                "median_template_score": round(
                    float(np.median([float(item["best_score"]) for item in run])), 6
                ),
                "minimum_margin": round(min(float(item["margin"]) for item in run), 6),
            }
        )
        run_start = index
    return intervals


def _make_spotcheck(
    tokens: Sequence[dict[str, object]],
    crops: dict[float, np.ndarray],
    destination: Path,
    *,
    maximum_items: int = 36,
) -> list[str]:
    if not tokens:
        return []
    indices = np.linspace(0, len(tokens) - 1, min(maximum_items, len(tokens)), dtype=int)
    chosen = [tokens[int(index)] for index in indices]
    tile_width, tile_height = 400, 180
    columns = 4
    rows = math.ceil(len(chosen) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    token_ids: list[str] = []
    for position, token in enumerate(chosen):
        token_ids.append(str(token["token_id"]))
        time_s = round(float(token["asr_rough_time_s"]), 6)
        crop = Image.fromarray(crops[time_s], mode="RGB").resize((380, 140), Image.Resampling.NEAREST)
        x = (position % columns) * tile_width + 10
        y = (position // columns) * tile_height
        sheet.paste(crop, (x, y + 32))
        draw.text(
            (x, y + 8),
            f"{token['token_id']}  {float(token['asr_rough_time_s']):.2f}s  {token['speaker_id']}",
            fill="black",
        )
    sheet.save(destination)
    return token_ids


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    video_path = args.video.expanduser().resolve()
    words_json = args.words_json.expanduser().resolve()
    references_json = args.references_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = CropGeometry(*args.label_crop)
    media_duration_s = _media_duration_s(video_path)
    end_s = _resolve_analysis_end(args.end, media_duration_s=media_duration_s)
    if not math.isfinite(args.start) or args.start < 0.0 or args.start >= end_s:
        raise ValueError("analysis start must be finite, non-negative, and before end")
    references = _load_references(references_json)
    for speaker_id, reference in references.items():
        reference_time_s = float(reference["time_s"])
        if reference_time_s >= media_duration_s:
            raise ValueError(
                f"reference {speaker_id!r} at {reference_time_s:.6f}s lies outside "
                f"the {media_duration_s:.6f}s media"
            )
    speaker_groups = {
        speaker_id: str(reference["group"])
        for speaker_id, reference in references.items()
    }
    display_names = {
        speaker_id: str(reference["display_name"])
        for speaker_id, reference in references.items()
    }
    tokens, reading_audit = _load_tokens(
        words_json,
        start_s=args.start,
        end_s=end_s,
        require_complete_readings=args.require_complete_readings,
    )
    reference_times = [float(value["time_s"]) for value in references.values()]
    vote_offsets = (-args.vote_offset, 0.0, args.vote_offset)
    guard_offsets = (-args.guard_offset, args.guard_offset)
    targets = _target_times(
        start_s=args.start,
        end_s=end_s,
        sample_hz=args.sample_hz,
        reference_times=reference_times,
        tokens=tokens,
        vote_offsets_s=vote_offsets,
        guard_offsets_s=guard_offsets,
    )
    crops, video_metadata = _decode_label_crops(video_path, targets, geometry)
    templates = {
        speaker: _label_feature(crops[round(float(info["time_s"]), 6)])
        for speaker, info in references.items()
    }

    timeline_samples: list[dict[str, object]] = []
    step = 1.0 / args.sample_hz
    for time_s in np.arange(args.start, end_s, step, dtype=np.float64):
        key = round(float(time_s), 6)
        result = classify_label(
            crops[key],
            templates,
            minimum_score=args.minimum_score,
            minimum_margin=args.minimum_margin,
        )
        timeline_samples.append(
            {
                "time_s": key,
                **result,
                "group": speaker_groups.get(str(result["speaker_id"]), "exclude"),
            }
        )

    for token in tokens:
        center = float(token["asr_rough_time_s"])
        observations: list[dict[str, object]] = []
        for offset in vote_offsets:
            time_s = round(center + offset, 6)
            if not (args.start <= time_s < end_s):
                continue
            result = classify_label(
                crops[time_s],
                templates,
                minimum_score=args.minimum_score,
                minimum_margin=args.minimum_margin,
            )
            observations.append({"offset_s": offset, **result})
        majority_speaker, unique_winner = _majority_vote(observations)
        speaker_id = _unanimous_speaker(observations)
        token["speaker_id"] = speaker_id
        token["group"] = speaker_groups.get(speaker_id, "exclude")
        token["speaker_display_name"] = display_names.get(speaker_id)
        token["speaker_majority_id"] = majority_speaker
        token["speaker_vote_unique"] = unique_winner
        token["speaker_vote_unanimous"] = speaker_id != "unknown"
        token["speaker_label_observations"] = observations
        guard_observations: list[dict[str, object]] = []
        for offset in guard_offsets:
            time_s = round(center + offset, 6)
            if not (args.start <= time_s < end_s):
                continue
            result = classify_label(
                crops[time_s],
                templates,
                minimum_score=args.minimum_score,
                minimum_margin=args.minimum_margin,
            )
            guard_observations.append({"offset_s": offset, **result})
        guarded_speakers = [str(item["speaker_id"]) for item in guard_observations]
        token["speaker_stable_for_analysis"] = (
            speaker_id != "unknown"
            and bool(guarded_speakers)
            and all(value == speaker_id for value in guarded_speakers)
        )
        token["speaker_guard_observations"] = guard_observations

    intervals = _compress_intervals(
        timeline_samples,
        start_s=args.start,
        end_s=end_s,
        sample_hz=args.sample_hz,
        speaker_groups=speaker_groups,
    )
    contact_sheet = output_dir / "speaker_label_spotcheck.png"
    spotcheck_token_ids = _make_spotcheck(tokens, crops, contact_sheet)

    speaker_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for token in tokens:
        speaker = str(token["speaker_id"])
        group = str(token["group"])
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        group_counts[group] = group_counts.get(group, 0) + 1
    unknown_samples = sum(item["speaker_id"] == "unknown" for item in timeline_samples)
    manifest = {
        "schema_version": 1,
        "scope": {
            "start_s": args.start,
            "end_s": end_s,
            "media_duration_s": media_duration_s,
            "end_source": "media_duration" if args.end is None else "command_line",
            "note": "Only material within the configured analysis window is included.",
        },
        "method": {
            "identity_signal": "configured on-screen display-name label crop only",
            "excluded_signals": ["face appearance", "voice identity", "accent", "nationality"],
            "label_crop_xyxy": list(geometry.box),
            "reference_labels": references,
            "reference_configuration": {
                "path": str(references_json),
                "sha256": _sha256(references_json),
            },
            "sample_hz": args.sample_hz,
            "vote_offsets_s": list(vote_offsets),
            "analysis_guard_offsets_s": list(guard_offsets),
            "minimum_dice_score": args.minimum_score,
            "minimum_winner_margin": args.minimum_margin,
            "asr_role": "token pre-selection only; not the final acoustic release time",
            "token_inventory_scope": reading_audit["token_inventory_scope"],
            "reading_requirement": (
                "complete analysis requires word-level reading/kana/pronunciation for every word"
            ),
        },
        "video": {"path": str(video_path), **video_metadata},
        "transcript": {"path": str(words_json), "reading_audit": reading_audit},
        "summary": {
            "timeline_sample_count": len(timeline_samples),
            "unknown_timeline_samples": unknown_samples,
            "unknown_timeline_fraction": unknown_samples / max(len(timeline_samples), 1),
            "interval_count": len(intervals),
            "bilabial_token_count": len(tokens),
            "token_counts_by_speaker": speaker_counts,
            "token_counts_by_group": group_counts,
            "stable_analysis_token_count": sum(
                bool(token["speaker_stable_for_analysis"]) for token in tokens
            ),
            "spotcheck_token_ids": spotcheck_token_ids,
        },
        "speaker_intervals": intervals,
        "tokens": tokens,
    }
    (output_dir / "speaker_token_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "speaker_timeline_samples.json").write_text(
        json.dumps(timeline_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "speaker_intervals.json").write_text(
        json.dumps(intervals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "bilabial_token_manifest.json").write_text(
        json.dumps(tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1")
    return parts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("words_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--references-json",
        required=True,
        type=Path,
        help=(
            "JSON mapping speaker_id to {display_name, time_s, group}; "
            "group must be candidate, control, or exclude"
        ),
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="analysis end in seconds (default: probed media duration)",
    )
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--vote-offset", type=float, default=0.10)
    parser.add_argument(
        "--guard-offset",
        type=float,
        default=0.75,
        help="Require the same displayed label this many seconds before and after a token",
    )
    parser.add_argument("--label-crop", type=_parse_crop, default=(10, 1000, 200, 1070))
    parser.add_argument("--minimum-score", type=float, default=0.55)
    parser.add_argument("--minimum-margin", type=float, default=0.12)
    parser.add_argument(
        "--require-complete-readings",
        action="store_true",
        help=(
            "fail unless every in-scope ASR word has reading, kana, or pronunciation; "
            "required for a complete /p,b/ inventory"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
