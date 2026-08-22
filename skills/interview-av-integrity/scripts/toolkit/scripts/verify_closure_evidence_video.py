#!/usr/bin/env python3
"""Mechanically verify a rendered lip-closure evidence video.

The effective render manifest is the sole authority for event IDs, event
count, source timestamps, classifications, card lengths, output timing, and
slow-frame reuse. The checks establish reproducibility of the edit and
encode; they do not judge lip movement or infer the cause of an A/V mismatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import av
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from video_integrity_analyzer.layout_reference import (  # noqa: E402
    LayoutReferenceError,
    validate_effective_layout_reference,
    validate_effective_layout_review,
)


ALLOWED_CLASSIFICATIONS = ("closure_absent", "contact_reference", "sync_reference")
FRAME_MAP_FIELDS = {
    "output_frame_index", "output_pts_s", "case_order", "event_id",
    "phase", "source_pts_s", "source_frame_index",
}


class VerificationError(RuntimeError):
    """Raised for a failed QA invariant."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: dict[str, Any]
    error: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} is not numeric: {value!r}") from error
    _require(math.isfinite(result), f"{label} is not finite: {value!r}")
    return result


def _integer(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} is not an integer: {value!r}") from error
    _require(isinstance(value, int) or str(value).strip() == str(result),
             f"{label} is not an exact integer: {value!r}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(payload, dict), "manifest root must be a JSON object")
    return payload


def _render_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    render = manifest.get("render")
    _require(isinstance(render, dict), "manifest.render must be an object")
    width = _integer(render.get("width"), "render.width")
    height = _integer(render.get("height"), "render.height")
    fps = _finite_float(render.get("fps"), "render.fps")
    audio_rate = _integer(render.get("audio_sample_rate"), "render.audio_sample_rate")
    repeat = _integer(render.get("slow_source_frame_repeat"), "render.slow_source_frame_repeat")
    _require(width > 0 and height > 0, "render dimensions must be positive")
    _require(fps > 0.0, "render.fps must be positive")
    _require(audio_rate > 0, "render.audio_sample_rate must be positive")
    _require(repeat >= 2, "render.slow_source_frame_repeat must be at least 2")
    _require(
        _integer(render.get("normal_source_frames_per_output_frame"),
                 "render.normal_source_frames_per_output_frame") == 1,
        "normal pass must map one source frame to one output frame",
    )
    _require(render.get("video_interpolation") is False,
             "render.video_interpolation must be false")
    frame_timing = render.get("source_frame_timing")
    _require(isinstance(frame_timing, dict),
             "render.source_frame_timing must be an object")
    _require(frame_timing.get("is_cfr") is True,
             "source frame timing was not verified as strict CFR")
    timing_fps = _finite_float(
        frame_timing.get("expected_fps"), "render.source_frame_timing.expected_fps"
    )
    _require(abs(timing_fps - fps) <= 1e-9,
             "source frame timing FPS disagrees with render FPS")

    limitation = str(render.get("limitation", "")).strip()
    _require(limitation, "render.limitation must be non-empty")
    folded = limitation.casefold()
    required_concepts = {
        "cause": ("原因", "cause"),
        "identity": ("本人", "identity"),
        "nationality": ("国籍", "nationality"),
        "affiliation": ("所属", "affiliation"),
        "intent": ("意図", "intent"),
        "non-determination": (
            "判定するものではありません", "does not determine", "not determine",
        ),
    }
    for concept, alternatives in required_concepts.items():
        _require(any(token.casefold() in folded for token in alternatives),
                 f"render.limitation does not disclaim {concept}")

    layout = render.get("layout")
    _require(isinstance(layout, dict), "render.layout must be an object")
    inset = layout.get("mouth_inset_xyxy")
    _require(isinstance(inset, list) and len(inset) == 4,
             "render.layout.mouth_inset_xyxy must contain four coordinates")
    inset_xyxy = tuple(_integer(value, "render.layout.mouth_inset_xyxy") for value in inset)
    x0, y0, x1, y1 = inset_xyxy
    _require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height,
             f"render mouth inset is outside the output frame: {inset_xyxy}")
    try:
        layout_reference = validate_effective_layout_reference(
            render.get("layout_reference")
        )
        input_manifest_value = manifest.get("input_event_manifest")
        if not isinstance(input_manifest_value, str) or not input_manifest_value.strip():
            raise LayoutReferenceError(
                "input_event_manifest is required for layout approval verification"
            )
        layout_review = validate_effective_layout_review(
            render.get("layout_review"),
            Path(input_manifest_value).expanduser().resolve(),
            layout_reference,
        )
    except LayoutReferenceError as error:
        raise VerificationError(str(error)) from error
    return {
        "width": width, "height": height, "fps": fps,
        "audio_rate": audio_rate, "slow_repeat": repeat,
        "frame_s": 1.0 / fps, "mouth_inset_xyxy": inset_xyxy,
        "limitation": limitation,
        "layout_reference": layout_reference,
        "layout_review": layout_review,
    }


def _card_spec(manifest: dict[str, Any], fps: float) -> dict[str, int]:
    cards = manifest.get("cards")
    _require(isinstance(cards, dict), "manifest.cards must be an object")
    values: dict[str, int] = {}
    for key in ("intro_frames", "event_gap_frames", "outro_frames"):
        value = _integer(cards.get(key), f"cards.{key}")
        _require(value >= 0, f"cards.{key} must be non-negative")
        values[key] = value
    for frame_key, duration_key in (
        ("intro_frames", "intro_duration_s"),
        ("event_gap_frames", "event_gap_duration_s"),
        ("outro_frames", "outro_duration_s"),
    ):
        duration = _finite_float(cards.get(duration_key), f"cards.{duration_key}")
        _require(abs(duration - values[frame_key] / fps) <= 1e-6,
                 f"cards.{duration_key} disagrees with {frame_key}")
    return values


def _normalized_roi(case: dict[str, Any], event_id: str) -> tuple[float, float, float, float]:
    raw = case.get("mouth_crop_normalized_xyxy")
    _require(isinstance(raw, list) and len(raw) == 4,
             f"{event_id}.mouth_crop_normalized_xyxy must contain four coordinates")
    roi = tuple(_finite_float(value, f"{event_id}.mouth_crop_normalized_xyxy") for value in raw)
    x0, y0, x1, y1 = roi
    _require(0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0,
             f"{event_id}: mouth ROI is outside the normalized frame: {roi}")
    return roi


def verify_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every event and derive the complete output timeline."""

    manifest = load_json_object(path)
    spec = _render_spec(manifest)
    fps = float(spec["fps"])
    repeat = int(spec["slow_repeat"])
    cards = _card_spec(manifest, fps)
    cases = manifest.get("cases")
    _require(isinstance(cases, list) and cases,
             "manifest.cases must be a non-empty list")
    _require(all(isinstance(case, dict) for case in cases),
             "every case must be an object")
    ids = [str(case.get("event_id", "")).strip() for case in cases]
    _require(all(ids), "event IDs must be non-empty")
    _require(len(set(ids)) == len(ids), f"event IDs are not unique: {ids}")

    releases = [
        _finite_float(case.get("source_release_s"), f"{event_id}.source_release_s")
        for event_id, case in zip(ids, cases)
    ]
    _require(all(right >= left for left, right in zip(releases, releases[1:])),
             f"case order is not chronological by source_release_s: {releases}")

    actual_counts: Counter[str] = Counter()
    cursor = cards["intro_frames"]
    phase_frame_counts: dict[str, dict[str, int]] = {}
    for expected_order, (event_id, release, case) in enumerate(
        zip(ids, releases, cases), start=1
    ):
        _require(_integer(case.get("order"), f"{event_id}.order") == expected_order,
                 f"{event_id}: order must be {expected_order}")
        classification = str(case.get("classification", ""))
        _require(classification in ALLOWED_CLASSIFICATIONS,
                 f"{event_id}: unsupported classification {classification!r}")
        actual_counts[classification] += 1
        roi = _normalized_roi(case, event_id)
        pixel_roi = case.get("mouth_crop_xywh")
        if pixel_roi is not None:
            _require(isinstance(pixel_roi, list) and len(pixel_roi) == 4,
                     f"{event_id}.mouth_crop_xywh must contain four coordinates")
            got = tuple(_integer(value, f"{event_id}.mouth_crop_xywh") for value in pixel_roi)
            expected = (
                round(roi[0] * int(spec["width"])),
                round(roi[1] * int(spec["height"])),
                round((roi[2] - roi[0]) * int(spec["width"])),
                round((roi[3] - roi[1]) * int(spec["height"])),
            )
            _require(got == expected,
                     f"{event_id}: pixel mouth ROI {got} != normalized ROI {expected}")

        phases: dict[str, dict[str, float | int]] = {}
        for phase_name in ("normal", "slow"):
            raw = case.get(phase_name)
            _require(isinstance(raw, dict),
                     f"{event_id}.{phase_name} must be an object")
            source_start = _finite_float(raw.get("source_start_s"),
                                         f"{event_id}.{phase_name}.source_start_s")
            source_end = _finite_float(raw.get("source_end_s"),
                                       f"{event_id}.{phase_name}.source_end_s")
            output_start = _finite_float(raw.get("output_start_s"),
                                         f"{event_id}.{phase_name}.output_start_s")
            output_end = _finite_float(raw.get("output_end_s"),
                                       f"{event_id}.{phase_name}.output_end_s")
            speed = _finite_float(raw.get("speed"), f"{event_id}.{phase_name}.speed")
            marker = _finite_float(raw.get("marker_output_s"),
                                   f"{event_id}.{phase_name}.marker_output_s")
            frame_count = _integer(raw.get("output_frame_count"),
                                   f"{event_id}.{phase_name}.output_frame_count")
            expected_speed = 1.0 if phase_name == "normal" else 1.0 / repeat
            _require(abs(speed - expected_speed) <= 1e-12,
                     f"{event_id}.{phase_name}: speed {speed} != {expected_speed}")
            _require(source_start < release < source_end,
                     f"{event_id}.{phase_name}: source window does not straddle release")
            expected_count = int(round((source_end - source_start) / speed * fps))
            _require(expected_count > 0 and frame_count == expected_count,
                     f"{event_id}.{phase_name}: frame count {frame_count} != derived {expected_count}")
            _require(abs((output_end - output_start) - frame_count / fps) <= 1e-6,
                     f"{event_id}.{phase_name}: output duration does not match frame count")
            _require(abs(output_start - cursor / fps) <= 1e-6,
                     f"{event_id}.{phase_name}: output start does not match declared timeline")
            expected_marker = output_start + (release - source_start) / speed
            _require(abs(marker - expected_marker) <= 1e-6,
                     f"{event_id}.{phase_name}: marker {marker:.6f} differs from formula {expected_marker:.6f}")
            phases[phase_name] = {
                "source_start_s": source_start,
                "source_end_s": source_end,
                "output_frame_count": frame_count,
            }
            cursor += frame_count

        normal, slow = phases["normal"], phases["slow"]
        _require(abs(float(normal["source_start_s"]) - float(slow["source_start_s"])) <= 1e-6,
                 f"{event_id}: normal/slow source starts differ")
        _require(abs(float(normal["source_end_s"]) - float(slow["source_end_s"])) <= 1e-6,
                 f"{event_id}: normal/slow source ends differ")
        _require(int(slow["output_frame_count"]) == repeat * int(normal["output_frame_count"]),
                 f"{event_id}: slow frame count is not {repeat}x normal")
        phase_frame_counts[event_id] = {
            "normal": int(normal["output_frame_count"]),
            "slow": int(slow["output_frame_count"]),
        }
        cursor += cards["event_gap_frames"]

    declared_counts = manifest.get("classification_counts")
    _require(isinstance(declared_counts, dict),
             "manifest.classification_counts must be an object")
    _require(set(declared_counts) == set(ALLOWED_CLASSIFICATIONS),
             f"classification_counts keys must be {list(ALLOWED_CLASSIFICATIONS)}")
    for classification in ALLOWED_CLASSIFICATIONS:
        declared = _integer(declared_counts.get(classification),
                            f"classification_counts.{classification}")
        _require(declared == actual_counts[classification],
                 f"classification_counts.{classification} {declared} != actual {actual_counts[classification]}")

    cursor += cards["outro_frames"]
    output = manifest.get("output")
    _require(isinstance(output, dict), "manifest.output must be an object")
    total_frames = _integer(output.get("total_frame_count"),
                            "output.total_frame_count")
    expected_duration = _finite_float(output.get("expected_duration_s"),
                                      "output.expected_duration_s")
    _require(total_frames == cursor,
             f"output.total_frame_count {total_frames} != timeline-derived {cursor}")
    _require(abs(expected_duration - total_frames / fps) <= 1e-6,
             "output.expected_duration_s disagrees with total frames and fps")
    return manifest, {
        "case_count": len(cases),
        "classification_counts": dict(actual_counts),
        "expected_duration_s": expected_duration,
        "total_frame_count": total_frames,
        "event_order": ids,
        "source_release_order_s": releases,
        "cards": cards,
        "phase_frame_counts": phase_frame_counts,
        "render": spec,
        "marker_tolerance_s": 1e-6,
    }


def _read_frame_map(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            _require(FRAME_MAP_FIELDS <= fields,
                     f"frame map lacks columns: {sorted(FRAME_MAP_FIELDS - fields)}")
            return list(reader)
    except OSError as error:
        raise VerificationError(f"cannot read frame map {path}: {error}") from error


def _blank_source(row: dict[str, str]) -> bool:
    return not row["source_pts_s"].strip() and not row["source_frame_index"].strip()


def verify_frame_map(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check CFR timing and exact manifest-declared native-frame reuse."""

    spec = _render_spec(manifest)
    fps = float(spec["fps"])
    frame_s = float(spec["frame_s"])
    repeat = int(spec["slow_repeat"])
    cards = _card_spec(manifest, fps)
    rows = _read_frame_map(path)
    expected_total = _integer(manifest["output"]["total_frame_count"],
                              "output.total_frame_count")
    _require(len(rows) == expected_total,
             f"frame map rows {len(rows)} != manifest total frames {expected_total}")
    indices = [_integer(row["output_frame_index"], "output_frame_index") for row in rows]
    _require(indices == list(range(len(rows))),
             "output_frame_index is not contiguous from zero")
    output_pts = [_finite_float(row["output_pts_s"], "output_pts_s") for row in rows]
    max_pts_error = max((abs(pts - index / fps)
                         for index, pts in enumerate(output_pts)), default=0.0)
    _require(max_pts_error <= 1e-6,
             f"output PTS is not exact {fps:g} fps CFR (max error {max_pts_error:.9f}s)")
    deltas = [right - left for left, right in zip(output_pts, output_pts[1:])]
    _require(all(delta > 0 for delta in deltas), "output PTS is not strictly increasing")
    max_step_error = max((abs(delta - frame_s) for delta in deltas), default=0.0)
    _require(max_step_error <= 1e-6,
             f"output frame interval differs from 1/{fps:g} (max error {max_step_error:.9f}s)")
    phases = Counter(row["phase"] for row in rows)
    allowed_phases = {"intro", "normal", "slow", "gap", "outro"}
    _require(not (set(phases) - allowed_phases),
             f"unexpected frame-map phases: {sorted(set(phases) - allowed_phases)}")

    def require_card(start: int, count: int, phase: str) -> None:
        selected = rows[start:start + count]
        _require(len(selected) == count, f"{phase}: truncated frame-map section")
        _require(all(row["phase"] == phase and not row["event_id"].strip()
                     and not row["case_order"].strip() and _blank_source(row)
                     for row in selected),
                 f"{phase}: row metadata differs from manifest card section")

    cursor = 0
    require_card(cursor, cards["intro_frames"], "intro")
    cursor += cards["intro_frames"]
    normal_rows_total = 0
    slow_rows_total = 0
    for case in manifest["cases"]:
        event_id = str(case["event_id"])
        order = _integer(case["order"], f"{event_id}.order")
        phase_data: dict[str, tuple[list[float], list[int]]] = {}
        for phase_name in ("normal", "slow"):
            phase = case[phase_name]
            count = _integer(phase["output_frame_count"],
                             f"{event_id}.{phase_name}.output_frame_count")
            selected = rows[cursor:cursor + count]
            _require(len(selected) == count,
                     f"{event_id}/{phase_name}: truncated frame-map section")
            _require(all(row["phase"] == phase_name and row["event_id"] == event_id
                         and _integer(row["case_order"], "case_order") == order
                         for row in selected),
                     f"{event_id}/{phase_name}: row metadata differs from manifest")
            source_pts = [
                _finite_float(row["source_pts_s"], f"{event_id}/{phase_name}.source_pts_s")
                for row in selected
            ]
            source_indices = [
                _integer(row["source_frame_index"], f"{event_id}/{phase_name}.source_frame_index")
                for row in selected
            ]
            phase_data[phase_name] = (source_pts, source_indices)
            cursor += count

        normal_pts, normal_indices = phase_data["normal"]
        slow_pts, slow_indices = phase_data["slow"]
        normal_rows_total += len(normal_indices)
        slow_rows_total += len(slow_indices)
        _require(all(right == left + 1
                     for left, right in zip(normal_indices, normal_indices[1:])),
                 f"{event_id}/normal: source frames are not consecutive 1:1")
        source_step_tolerance = max(0.002, frame_s * 0.05)
        _require(all(abs((right - left) - frame_s) <= source_step_tolerance
                     for left, right in zip(normal_pts, normal_pts[1:])),
                 f"{event_id}/normal: source PTS is not approximately one source frame apart")
        source_start = _finite_float(case["normal"]["source_start_s"], "source_start_s")
        nearest_tolerance = frame_s / 2.0 + 0.001
        _require(max((abs(value - (source_start + index * frame_s))
                      for index, value in enumerate(normal_pts)), default=0.0)
                 <= nearest_tolerance,
                 f"{event_id}/normal: native-frame selection is not nearest to the discrete target")
        _require(len(slow_indices) == len(normal_indices) * repeat,
                 f"{event_id}/slow: frame count is not {repeat} per normal source frame")
        for native_offset, (normal_index, normal_time) in enumerate(
            zip(normal_indices, normal_pts)
        ):
            start = native_offset * repeat
            _require(slow_indices[start:start + repeat] == [normal_index] * repeat,
                     f"{event_id}/slow: source frame {normal_index} is not held for exactly {repeat} outputs")
            _require(all(abs(value - normal_time) <= 1e-9
                         for value in slow_pts[start:start + repeat]),
                     f"{event_id}/slow: repeated source PTS differs from normal pass")

        for phase_name, (phase_pts, _) in phase_data.items():
            phase = case[phase_name]
            marker = _finite_float(phase["marker_output_s"], "marker_output_s")
            start_frame = int(round(
                _finite_float(phase["output_start_s"], "output_start_s") * fps
            ))
            marker_local = min(range(len(phase_pts)), key=lambda index: abs(
                output_pts[start_frame + index] - marker
            ))
            release = _finite_float(case["source_release_s"], "source_release_s")
            _require(abs(phase_pts[marker_local] - release) <= frame_s + 0.001,
                     f"{event_id}/{phase_name}: nearest marker frame does not map near release")

        gap_count = cards["event_gap_frames"]
        gap_rows = rows[cursor:cursor + gap_count]
        _require(len(gap_rows) == gap_count,
                 f"{event_id}/gap: truncated frame-map section")
        _require(all(row["phase"] == "gap" and row["event_id"] == event_id
                     and _integer(row["case_order"], "case_order") == order
                     for row in gap_rows),
                 f"{event_id}/gap: row metadata differs from manifest")
        if gap_rows:
            _require(all(
                _integer(row["source_frame_index"], "gap.source_frame_index") == normal_indices[-1]
                and abs(_finite_float(row["source_pts_s"], "gap.source_pts_s") - normal_pts[-1]) <= 1e-9
                for row in gap_rows
            ), f"{event_id}/gap: hold frame differs from final normal source frame")
        cursor += gap_count

    require_card(cursor, cards["outro_frames"], "outro")
    cursor += cards["outro_frames"]
    _require(cursor == len(rows),
             f"frame-map timeline consumed {cursor} of {len(rows)} rows")
    return {
        "row_count": len(rows),
        "first_output_pts_s": output_pts[0] if output_pts else None,
        "last_output_pts_s": output_pts[-1] if output_pts else None,
        "median_output_frame_step_s": statistics.median(deltas) if deltas else None,
        "max_output_pts_error_s": max_pts_error,
        "normal_frame_rows": normal_rows_total,
        "slow_frame_rows": slow_rows_total,
        "slow_source_frame_repeat": repeat,
        "phase_counts": dict(phases),
    }


def verify_media(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fully decode the MP4 and compare it with the manifest render spec."""

    spec = _render_spec(manifest)
    fps = float(spec["fps"])
    frame_s = float(spec["frame_s"])
    _require(path.suffix.lower() == ".mp4",
             f"video container extension is not .mp4: {path.name}")
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as error:
        raise VerificationError(f"cannot open video: {error}") from error
    with container:
        videos = [stream for stream in container.streams if stream.type == "video"]
        audios = [stream for stream in container.streams if stream.type == "audio"]
        others = [stream for stream in container.streams
                  if stream.type not in {"video", "audio"}]
        _require(len(videos) == 1, f"expected one video stream, found {len(videos)}")
        _require(len(audios) == 1, f"expected one audio stream, found {len(audios)}")
        _require(not others,
                 f"unexpected streams: {[(stream.index, stream.type) for stream in others]}")
        video_stream, audio_stream = videos[0], audios[0]
        _require(video_stream.codec_context.name == "h264", "video codec must be h264")
        _require(
            (video_stream.codec_context.width, video_stream.codec_context.height)
            == (spec["width"], spec["height"]),
            "video dimensions differ from manifest render dimensions",
        )
        rate = video_stream.average_rate
        _require(rate is not None and abs(float(rate) - fps) <= 1e-6,
                 f"average video rate is {rate}, expected {fps:g}")
        _require(audio_stream.codec_context.name == "aac", "audio codec must be aac")
        _require(audio_stream.codec_context.sample_rate == spec["audio_rate"],
                 "audio sample rate differs from manifest")

        video_times: list[float] = []
        decoded_pix_fmts: set[str] = set()
        audio_first: float | None = None
        audio_last: float | None = None
        audio_frame_count = 0
        audio_sample_count = 0
        audio_energy = 0.0
        audio_peak = 0.0
        try:
            for packet in container.demux():
                for frame in packet.decode():
                    if packet.stream.type == "video":
                        _require(frame.pts is not None,
                                 "decoded video frame has no PTS")
                        video_times.append(float(frame.pts * frame.time_base))
                        decoded_pix_fmts.add(frame.format.name)
                    elif packet.stream.type == "audio":
                        _require(frame.pts is not None and frame.sample_rate > 0,
                                 "decoded audio frame has invalid timing")
                        start = float(frame.pts * frame.time_base)
                        end = start + frame.samples / frame.sample_rate
                        audio_first = start if audio_first is None else min(audio_first, start)
                        audio_last = end if audio_last is None else max(audio_last, end)
                        raw = frame.to_ndarray()
                        values = raw.astype(np.float64, copy=False)
                        _require(bool(np.isfinite(values).all()),
                                 "decoded audio contains NaN or infinity")
                        if np.issubdtype(raw.dtype, np.integer):
                            values /= max(float(np.iinfo(raw.dtype).max), 1.0)
                        audio_energy += float(np.square(values).sum())
                        audio_peak = max(audio_peak,
                                         float(np.abs(values).max(initial=0.0)))
                        audio_sample_count += int(values.size)
                        audio_frame_count += 1
        except (av.error.FFmpegError, OSError, ValueError) as error:
            raise VerificationError(f"full media decode failed: {error}") from error

        _require(video_times, "no video frames decoded")
        _require(audio_first is not None and audio_last is not None
                 and audio_frame_count > 0,
                 "no timestamped audio frames decoded")
        _require(decoded_pix_fmts == {"yuv420p"},
                 f"decoded pixel format is {sorted(decoded_pix_fmts)}, expected yuv420p")
        _require(all(right > left for left, right in zip(video_times, video_times[1:])),
                 "decoded video PTS is not strictly increasing")
        max_step_error = max((abs((right - left) - frame_s)
                              for left, right in zip(video_times, video_times[1:])),
                             default=0.0)
        _require(max_step_error <= 1e-4,
                 f"decoded video is not {fps:g} fps CFR (max step error {max_step_error:.9f}s)")
        expected_duration = _finite_float(manifest["output"]["expected_duration_s"],
                                          "expected_duration_s")
        expected_frames = _integer(manifest["output"]["total_frame_count"],
                                   "total_frame_count")
        _require(len(video_times) == expected_frames,
                 f"decoded video frames {len(video_times)} != manifest {expected_frames}")
        video_start = video_times[0]
        video_end = video_times[-1] + frame_s
        video_duration = video_end - video_start
        _require(abs(video_duration - expected_duration) <= frame_s,
                 f"decoded duration {video_duration:.6f}s != manifest {expected_duration:.6f}s")
        edge_tolerance = max(0.050, frame_s + 0.001)
        _require(abs(video_start - audio_first) <= edge_tolerance,
                 "audio/video starts differ too much")
        _require(abs(video_end - audio_last) <= edge_tolerance,
                 "audio/video ends differ too much")
        _require(audio_sample_count > 0, "decoded audio contains no samples")
        audio_rms = math.sqrt(audio_energy / audio_sample_count)
        _require(math.isfinite(audio_rms), "decoded audio RMS is not finite")
        return {
            "video_codec": video_stream.codec_context.name,
            "width": video_stream.codec_context.width,
            "height": video_stream.codec_context.height,
            "pixel_format": next(iter(decoded_pix_fmts)),
            "fps": float(rate),
            "decoded_video_frames": len(video_times),
            "video_start_s": video_start,
            "video_end_s": video_end,
            "video_duration_s": video_duration,
            "max_video_frame_step_error_s": max_step_error,
            "audio_codec": audio_stream.codec_context.name,
            "audio_sample_rate": audio_stream.codec_context.sample_rate,
            "decoded_audio_frames": audio_frame_count,
            "audio_start_s": audio_first,
            "audio_end_s": audio_last,
            "audio_rms": audio_rms,
            "audio_peak": audio_peak,
            "av_start_difference_s": abs(video_start - audio_first),
            "av_end_difference_s": abs(video_end - audio_last),
        }


def _run_check(name: str, function: Callable[[], dict[str, Any]]) -> CheckResult:
    try:
        return CheckResult(name=name, passed=True, details=function())
    except Exception as error:
        return CheckResult(name=name, passed=False, details={},
                           error=f"{type(error).__name__}: {error}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_reports(
    output_dir: Path, *, paths: dict[str, Path], hashes: dict[str, str],
    checks: list[CheckResult],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = all(check.passed for check in checks)
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "generated_at": generated,
        "scope_note": (
            "Mechanical render/encode QA only; this result does not determine the cause "
            "of an A/V mismatch or a speaker's identity, nationality, affiliation, or intent."
        ),
        "inputs": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": hashes,
        "checks": [asdict(check) for check in checks],
    }
    json_path = output_dir / "closure_evidence_verification.json"
    json_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 閉鎖証拠動画 QA 結果", "",
        f"- 結果: **{'PASS' if passed else 'FAIL'}**",
        f"- 実施日時: `{generated}`",
        "- 範囲: 時間写像・フレーム対応・MP4符号化の機械検証のみ。A/V差の原因や本人性、国籍、所属、意図は判定しない。",
        "", "## 検証項目", "",
    ]
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"- **{mark}** `{check.name}`")
        if check.error:
            lines.append(f"  - {check.error}")
        elif check.details:
            compact = json.dumps(_json_safe(check.details), ensure_ascii=False,
                                 sort_keys=True)
            lines.append(f"  - `{compact}`")
    lines.extend(["", "## SHA-256", ""])
    for name, digest in hashes.items():
        lines.append(f"- `{digest}`  `{paths[name].resolve()}`")
    lines.extend(["", f"機械検証: **{'PASS' if passed else 'FAIL'}**", ""])
    markdown_path = output_dir / "QA_RESULTS.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, json_path


def verify_all(
    *, output_dir: Path, manifest_path: Path, frame_map_path: Path,
    video_path: Path,
) -> tuple[bool, list[CheckResult], dict[str, str], tuple[Path, Path]]:
    paths = {"manifest": manifest_path, "frame_map": frame_map_path,
             "video": video_path}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    input_check = CheckResult(
        name="input_files", passed=not missing,
        details={"files": {name: str(path.resolve())
                           for name, path in paths.items()}},
        error=None if not missing else f"missing files: {missing}",
    )
    checks = [input_check]
    hashes = {name: sha256_file(path) for name, path in paths.items()
              if path.is_file()}
    manifest: dict[str, Any] | None = None
    if input_check.passed:
        try:
            manifest, details = verify_manifest(manifest_path)
            checks.append(CheckResult("manifest_declared_timeline_and_mapping",
                                      True, details))
        except Exception as error:
            checks.append(CheckResult(
                "manifest_declared_timeline_and_mapping", False, {},
                f"{type(error).__name__}: {error}",
            ))
    else:
        checks.append(CheckResult("manifest_declared_timeline_and_mapping",
                                  False, {}, "skipped: missing input"))
    if manifest is not None:
        checks.append(_run_check(
            "frame_map_cfr_and_source_reuse",
            lambda: verify_frame_map(frame_map_path, manifest),
        ))
        checks.append(_run_check(
            "mp4_streams_full_decode_and_duration",
            lambda: verify_media(video_path, manifest),
        ))
    else:
        checks.append(CheckResult("frame_map_cfr_and_source_reuse", False, {},
                                  "skipped: manifest failed"))
        checks.append(CheckResult("mp4_streams_full_decode_and_duration", False, {},
                                  "skipped: manifest failed"))
    reports = write_reports(output_dir, paths=paths, hashes=hashes,
                            checks=checks)
    return all(check.passed for check in checks), checks, hashes, reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory for QA_RESULTS.md and JSON")
    parser.add_argument("--manifest", type=Path,
                        help="default: OUTPUT_DIR/closure_evidence_manifest.json")
    parser.add_argument("--frame-map", type=Path,
                        help="default: OUTPUT_DIR/closure_evidence_frame_map.csv")
    parser.add_argument("--video", type=Path,
                        help="default: OUTPUT_DIR/closure_evidence_video.mp4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    manifest = (args.manifest or output_dir / "closure_evidence_manifest.json").expanduser().resolve()
    frame_map = (args.frame_map or output_dir / "closure_evidence_frame_map.csv").expanduser().resolve()
    video = (args.video or output_dir / "closure_evidence_video.mp4").expanduser().resolve()
    passed, checks, hashes, reports = verify_all(
        output_dir=output_dir, manifest_path=manifest,
        frame_map_path=frame_map, video_path=video,
    )
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}: {check.name}")
        if check.error:
            print(f"  {check.error}")
    for name, digest in hashes.items():
        print(f"SHA256 {name}: {digest}")
    print(f"Report: {reports[0]}")
    print(f"JSON: {reports[1]}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
