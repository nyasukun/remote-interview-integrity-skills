#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

RENDERING_DIR = Path(__file__).resolve().parent
if str(RENDERING_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERING_DIR))

from adapter import display_metrics
from common import (
    AUDIO_RATE,
    DEFAULT_LIMITATION,
    FPS,
    HEIGHT,
    LIMITATION_DETAIL,
    SAMPLES_PER_FRAME,
    WIDTH,
    apply_edge_fade,
    font,
    limitation_reference_strip,
    sha256,
)
from layout_reference import verify_approved_preview, verify_layout_reference
TOOLKIT_DIR = RENDERING_DIR.parent
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from acoustics.model import json_ready  # noqa: E402
from acoustics.signal_features import decode_audio  # noqa: E402
from acoustics.verify_artifacts import verify_acoustic_dir  # noqa: E402


class VerificationError(RuntimeError):
    pass


PROHIBITED_FIELD_NAMES = {
    "biometric_match",
    "closest_speaker",
    "composite_score",
    "distance",
    "identity_inference",
    "identity_probability",
    "identity_score",
    "match_probability",
    "ranking",
    "same_speaker_probability",
    "similarity",
    "similarity_score",
    "speaker_embedding",
    "speaker_identity_score",
    "speaker_similarity",
    "speaker_similarity_score",
    "voiceprint",
    "winner",
}

PROHIBITED_FIELD_FRAGMENTS = (
    "same_speaker",
    "speaker_similarity",
    "speaker_embedding",
    "voiceprint",
    "identity_probability",
    "match_probability",
    "biometric_match",
    "closest_speaker",
    "composite_score",
)

CANONICAL_LIMITATIONS = {
    DEFAULT_LIMITATION,
    "Descriptive signal differences only. No identity determination.",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _prohibited_field_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child_path = f"{path}.{key}"
            if normalized in PROHIBITED_FIELD_NAMES or any(
                fragment in normalized for fragment in PROHIBITED_FIELD_FRAGMENTS
            ):
                found.append(child_path)
            found.extend(_prohibited_field_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_prohibited_field_paths(child, f"{path}[{index}]"))
    return found


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested_metric(metrics: dict[str, Any], key: str) -> float | None:
    value: Any = metrics
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if value is not None else None


def _decode_video(path: Path, limitation: str) -> tuple[dict[str, Any], list[float], list[float]]:
    pts: list[float] = []
    errors: list[float] = []
    reference = limitation_reference_strip(limitation).astype(np.float32)
    with av.open(str(path), mode="r") as container:
        _require(len(container.streams.video) == 1, "output must contain one video stream")
        _require(len(container.streams.audio) == 1, "output must contain one audio stream")
        stream = container.streams.video[0]
        metadata = {
            "codec": str(stream.codec_context.name),
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "pixel_format": str(stream.codec_context.pix_fmt),
            "average_rate": float(stream.average_rate) if stream.average_rate else None,
            "comment": str(container.metadata.get("comment", "")),
        }
        for frame in container.decode(stream):
            _require(frame.pts is not None and frame.time_base is not None, "video frame lacks PTS")
            pts.append(float(frame.pts * frame.time_base))
            strip = frame.to_ndarray(format="rgb24")[HEIGHT - 84 :, :, :].astype(np.float32)
            errors.append(float(np.mean(np.abs(strip - reference))))
    return metadata, pts, errors


def _decode_audio(path: Path) -> tuple[dict[str, Any], np.ndarray, float]:
    blocks: list[np.ndarray] = []
    end_s = 0.0
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        metadata = {"codec": str(stream.codec_context.name), "sample_rate": int(stream.codec_context.sample_rate or 0), "channels": int(stream.codec_context.channels or 0)}
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                values = np.asarray(output.to_ndarray(), dtype=np.float32).reshape(2, -1)
                blocks.append(values)
                if output.pts is not None and output.time_base is not None:
                    end_s = max(end_s, float(output.pts * output.time_base) + values.shape[1] / AUDIO_RATE)
        for output in resampler.resample(None):
            values = np.asarray(output.to_ndarray(), dtype=np.float32).reshape(2, -1)
            blocks.append(values)
    audio = np.concatenate(blocks, axis=1) if blocks else np.zeros((2, 0), dtype=np.float32)
    return metadata, audio, end_s or audio.shape[1] / AUDIO_RATE


def _preservation(reference: np.ndarray, rendered: np.ndarray) -> dict[str, float | int]:
    reference_mono = np.mean(reference.astype(np.float64), axis=0)
    rendered_mono = np.mean(rendered.astype(np.float64), axis=0)
    decimation = 4
    ref = reference_mono[::decimation] - np.mean(reference_mono[::decimation])
    out = rendered_mono[::decimation] - np.mean(rendered_mono[::decimation])
    correlation = signal.correlate(out, ref, mode="full", method="fft")
    lags = signal.correlation_lags(out.size, ref.size, mode="full")
    allowed = np.abs(lags * decimation) <= 4096
    indices = np.flatnonzero(allowed)
    lag = int(lags[indices[int(np.argmax(correlation[allowed]))]] * decimation)
    if lag >= 0:
        ref_aligned = reference[:, : rendered.shape[1] - lag]
        out_aligned = rendered[:, lag : lag + ref_aligned.shape[1]]
    else:
        ref_aligned = reference[:, -lag:]
        out_aligned = rendered[:, : ref_aligned.shape[1]]
    count = min(ref_aligned.shape[1], out_aligned.shape[1])
    trim = min(960, max(0, count // 10))
    ref_aligned = ref_aligned[:, trim : count - trim].astype(np.float64)
    out_aligned = out_aligned[:, trim : count - trim].astype(np.float64)
    centered_ref = ref_aligned - np.mean(ref_aligned, axis=1, keepdims=True)
    centered_out = out_aligned - np.mean(out_aligned, axis=1, keepdims=True)
    denominator = float(np.linalg.norm(centered_ref) * np.linalg.norm(centered_out))
    normalized_correlation = float(np.sum(centered_ref * centered_out) / denominator) if denominator else 0.0
    ref_rms = float(np.sqrt(np.mean(ref_aligned**2)))
    out_rms = float(np.sqrt(np.mean(out_aligned**2)))
    rms_delta_db = 20 * math.log10(max(out_rms, 1e-12) / max(ref_rms, 1e-12))
    nrmse = float(np.sqrt(np.mean((out_aligned - ref_aligned) ** 2)) / max(ref_rms, 1e-12))
    return {"alignment_lag_samples": lag, "normalized_correlation": normalized_correlation, "rms_delta_db": rms_delta_db, "normalized_rmse": nrmse}


def _contact_sheet_targets(
    clips: list[dict[str, Any]], duration: float
) -> list[tuple[str, float]]:
    first_clip_start = min(float(clip["output_start_sample"]) for clip in clips) / AUDIO_RATE
    last_clip_end = max(float(clip["output_end_sample_exclusive"]) for clip in clips) / AUDIO_RATE
    targets: list[tuple[str, float]] = []
    if first_clip_start > 0.0:
        targets.append(("intro", first_clip_start / 2.0))
    targets.extend(
        (
            str(clip["clip_id"]),
            (
                float(clip["output_start_sample"])
                + float(clip["output_end_sample_exclusive"])
            )
            / (2 * AUDIO_RATE),
        )
        for clip in clips
    )
    if last_clip_end < duration:
        targets.append(("outro", last_clip_end + (duration - last_clip_end) / 2.0))
    return targets


def _contact_sheet(video: Path, clips: list[dict[str, Any]], duration: float, output: Path) -> None:
    targets = _contact_sheet_targets(clips, duration)
    final_frame = max(0, int(math.floor(duration * FPS)) - 1)
    wanted = [
        (min(final_frame, max(0, int(round(seconds * FPS)))), label)
        for label, seconds in targets
    ]
    wanted_indices = {index for index, _ in wanted}
    frames: dict[int, Image.Image] = {}
    with av.open(str(video), mode="r") as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in wanted_indices:
                frames[index] = frame.to_image().resize((480, 270), Image.Resampling.LANCZOS)
    columns = 3
    rows = math.ceil(len(wanted) / columns)
    sheet = Image.new("RGB", (1520, 58 + rows * 292), (5, 12, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 15), "Visual QA contact sheet — review required", font=font(24, True), fill=(232, 238, 243))
    for position, (index, label) in enumerate(wanted):
        row, column = divmod(position, columns)
        x, y = 20 + column * 500, 54 + row * 292
        if index in frames:
            sheet.paste(frames[index], (x, y))
        draw.rectangle((x, y, x + 480, y + 270), outline=(80, 96, 108), width=2)
        draw.text((x + 8, y + 6), f"{label}  t={index / FPS:.3f}s", font=font(13, True), fill=(232, 238, 243))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def verify(video: Path, effective: Path, frame_map: Path, audio_map: Path, output_dir: Path) -> dict[str, Any]:
    video, effective, frame_map, audio_map, output_dir = (path.expanduser().resolve() for path in (video, effective, frame_map, audio_map, output_dir))
    for path in (video, effective, frame_map, audio_map):
        _require(path.is_file(), f"missing input: {path}")
    payload = json.loads(effective.read_text(encoding="utf-8"))
    prohibited_paths = _prohibited_field_paths(payload)
    _require(
        not prohibited_paths,
        "prohibited field exists: " + ", ".join(prohibited_paths[:5]),
    )
    _require(effective.stat().st_size <= 262_144, "effective manifest is too large; PCM may be serialized")
    manifest = Path(payload["source_manifest"])
    _require(manifest.is_file() and sha256(manifest) == payload["source_manifest_sha256"], "source manifest hash mismatch")
    try:
        current_layout_reference = verify_layout_reference()
        current_layout_review = verify_approved_preview(manifest, current_layout_reference)
    except ValueError as error:
        raise VerificationError(f"layout reference/review verification failed: {error}") from error
    layout = payload.get("layout", {})
    _require(
        layout.get("reference_asset") == current_layout_reference,
        "effective layout reference provenance mismatch",
    )
    _require(
        layout.get("approved_preview") == current_layout_review,
        "effective approved-preview provenance mismatch",
    )
    _require(
        (layout.get("width"), layout.get("height"), layout.get("fps"), layout.get("pixel_format"))
        == (WIDTH, HEIGHT, FPS, "yuv420p"),
        "effective layout output profile mismatch",
    )
    clip_manifest = payload.get("clip_manifest", {})
    clip_manifest_path = Path(str(clip_manifest.get("path", "")))
    _require(clip_manifest_path.is_file() and sha256(clip_manifest_path) == clip_manifest.get("sha256"), "clip manifest hash mismatch")
    for entry in payload.get("provenance_manifests", []):
        path = Path(entry["path"])
        _require(path.is_file() and sha256(path) == entry["sha256"], f"provenance hash mismatch: {entry.get('role')}")
    comparison = payload.get("comparison", {})
    anchor_group = str(comparison.get("anchor_group", ""))
    _require(
        bool(anchor_group)
        and comparison.get("display_anchor_group") == anchor_group
        and comparison.get("point_group") == anchor_group,
        "declared/display/point anchor mismatch",
    )
    _require(comparison.get("delta_definition") == "group metric minus user-designated point metric", "signed delta definition mismatch")
    _require(comparison.get("shared_metric_scales") is True, "shared metric scales are not declared")
    summary = payload.get("summary", {})
    _require(summary.get("anchor_group") == anchor_group, "summary anchor mismatch")
    group_ids = {group["group_id"] for group in payload.get("groups", [])}
    role_by_group = {group["group_id"]: group.get("role") for group in payload.get("groups", [])}
    _require(2 <= len(group_ids) <= 4 and role_by_group.get(anchor_group) == "designated_anchor", "invalid anchor/group coverage")
    reference_groups = comparison.get("reference_groups", [])
    _require(1 <= len(reference_groups) <= 3, "comparison group count must be one to three")
    _require(set(reference_groups) == group_ids - {anchor_group}, "reference group coverage mismatch")
    for name, metric in summary.get("metrics", {}).items():
        deltas = metric.get("signed_deltas_to_designated", {})
        _require(set(deltas) == group_ids, f"{name}: delta coverage mismatch")
        _require(
            all(value is not None and abs(float(value)) <= 1e-12 for value in deltas[anchor_group].values()),
            f"{name}: designated anchor is not fixed at zero",
        )
    limitation = payload.get("limitation", {})
    _require(limitation.get("applies_to_every_frame") is True, "limitation is not declared permanent")
    limitation_text = str(limitation.get("permanent_text", ""))
    _require(
        limitation_text in CANONICAL_LIMITATIONS,
        "permanent limitation must equal an approved non-identity statement",
    )
    _require(
        limitation.get("detail") == LIMITATION_DETAIL,
        "limitation detail differs from the approved non-identity statement",
    )
    clips = payload.get("clips", [])
    _require(clips and sum(clip["group_id"] == anchor_group for clip in clips) == 1, "designated anchor clip count mismatch")
    for clip in clips:
        source = Path(clip["source"])
        _require(source.is_file() and sha256(source) == clip["source_sha256"], f"{clip['clip_id']}: source hash mismatch")

    authoritative = payload.get("authoritative_acoustic_match")
    _require(isinstance(authoritative, dict), "authoritative acoustic match report is missing")
    _require(authoritative.get("all_intervals_exact_match") is True, "authoritative intervals did not match")
    _require(authoritative.get("all_sources_exact_match") is True, "authoritative sources did not match")
    _require(authoritative.get("all_summaries_exact_match") is True, "authoritative summaries did not match")
    artifacts = authoritative.get("artifacts", {})
    for key in ("artifact_manifest", "acoustic_features"):
        entry = artifacts.get(key)
        _require(isinstance(entry, dict), f"authoritative {key} provenance is missing")
        path = Path(str(entry.get("path", "")))
        _require(path.is_file() and sha256(path) == entry.get("sha256"), f"authoritative {key} hash mismatch")
    artifact_manifest_path = Path(artifacts["artifact_manifest"]["path"])
    _require(verify_acoustic_dir(artifact_manifest_path.parent).get("status") == "PASS", "authoritative acoustic directory verification failed")
    acoustic_payload = json.loads(Path(artifacts["acoustic_features"]["path"]).read_text(encoding="utf-8"))
    _require(acoustic_payload.get("method", {}).get("config", {}).get("sample_rate") == AUDIO_RATE, "authoritative acoustic analysis rate is not 48 kHz")
    authoritative_by_id = {str(item.get("clip_id", "")): item for item in acoustic_payload.get("clips", [])}
    authoritative_groups = {str(item.get("group_id", "")): item for item in acoustic_payload.get("group_summaries", [])}
    effective_groups = {str(item.get("group_id", "")): item for item in payload.get("groups", [])}
    _require(set(authoritative_groups) == set(effective_groups), "authoritative group coverage mismatch")
    for group_id, group in effective_groups.items():
        expected_hex = "#" + "".join(f"{int(value):02X}" for value in group["color_rgb"])
        _require(authoritative_groups[group_id].get("group_label") == group.get("label"), f"{group_id}: authoritative group label mismatch")
        _require(str(authoritative_groups[group_id].get("display_color_hex", "")).upper() == expected_hex, f"{group_id}: authoritative group color mismatch")
    match_by_id = {str(item.get("clip_id", "")): item for item in authoritative.get("clips", [])}
    _require(set(authoritative_by_id) == set(match_by_id) == {clip["clip_id"] for clip in clips}, "authoritative clip coverage mismatch")
    for clip in clips:
        expected_summary = authoritative_by_id[clip["clip_id"]].get("summary")
        authoritative_clip = authoritative_by_id[clip["clip_id"]]
        authoritative_selection = authoritative_clip.get("selection", {})
        authoritative_source = authoritative_clip.get("source", {})
        _require(
            authoritative_clip.get("group_id") == clip.get("group_id"),
            f"{clip['clip_id']}: effective/authoritative group mismatch",
        )
        _require(
            abs(float(authoritative_selection.get("start_s")) - float(clip["source_start_s"])) <= 1e-9
            and abs(float(authoritative_selection.get("end_s")) - float(clip["source_end_s"])) <= 1e-9,
            f"{clip['clip_id']}: effective/authoritative interval mismatch",
        )
        _require(
            Path(str(authoritative_source.get("path", ""))).expanduser().resolve()
            == Path(clip["source"]).expanduser().resolve()
            and authoritative_source.get("sha256") == clip.get("source_sha256"),
            f"{clip['clip_id']}: effective/authoritative source mismatch",
        )
        _require(
            match_by_id[clip["clip_id"]].get("interval_exact_match") is True
            and match_by_id[clip["clip_id"]].get("source_exact_match") is True
            and match_by_id[clip["clip_id"]].get("summary_exact_match") is True,
            f"{clip['clip_id']}: authoritative match flags are not all true",
        )
        rendered_summary = clip.get("metrics", {}).get("authoritative_summary")
        _require(rendered_summary == expected_summary, f"{clip['clip_id']}: effective/authoritative summary mismatch")
        _require(match_by_id[clip["clip_id"]].get("canonical_summary_sha256") == _canonical_sha256(expected_summary), f"{clip['clip_id']}: canonical summary hash mismatch")
        expected_metrics = {**display_metrics(expected_summary), "authoritative_summary": expected_summary}
        _require(clip.get("metrics") == expected_metrics, f"{clip['clip_id']}: displayed metrics differ from authoritative registry mapping")
    for metric_name, metric in summary.get("metrics", {}).items():
        expected_rows: dict[str, Any] = {}
        for group_id in group_ids:
            values = [
                value
                for clip in clips
                if clip["group_id"] == group_id
                if (value := _nested_metric(clip["metrics"], metric_name)) is not None
            ]
            expected_rows[group_id] = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "median": float(np.median(values)) if values else None,
            }
        _require(metric.get("groups") == expected_rows, f"{metric_name}: group summary differs from authoritative clip metrics")
        anchor_value = expected_rows[anchor_group]["median"]
        expected_deltas = {
            group_id: {
                statistic: value - anchor_value if value is not None and anchor_value is not None else None
                for statistic, value in (("min", row["min"]), ("max", row["max"]), ("median", row["median"]))
            }
            for group_id, row in expected_rows.items()
        }
        _require(metric.get("signed_deltas_to_designated") == expected_deltas, f"{metric_name}: signed deltas differ from group minus designated")

    video_meta, pts, banner_errors = _decode_video(video, limitation_text)
    _require(video_meta["codec"] in {"h264", "libx264"}, "video is not H.264")
    _require((video_meta["width"], video_meta["height"], video_meta["pixel_format"]) == (WIDTH, HEIGHT, "yuv420p"), "video format mismatch")
    _require(video_meta["average_rate"] is not None and abs(video_meta["average_rate"] - FPS) < 1e-6, "video is not 24 fps")
    _require(limitation_text in video_meta["comment"], "container comment lacks limitation")
    expected_pts = np.arange(len(pts), dtype=np.float64) / FPS
    max_pts_error = float(np.max(np.abs(np.asarray(pts) - expected_pts)))
    _require(max_pts_error <= 1e-6, "video is not exact 24 fps CFR")
    _require(max(banner_errors) <= 14.0, "permanent limitation strip is missing or changed")
    frame_rows = _csv(frame_map)
    _require(len(frame_rows) == len(pts), "frame map count mismatch")
    timeline_payload = payload.get("timeline")
    _require(isinstance(timeline_payload, list) and timeline_payload, "effective timeline is missing")
    for index, row in enumerate(frame_rows):
        _require(int(row["output_frame_index"]) == index and abs(float(row["output_pts_s"]) - pts[index]) <= 1e-6, "frame map PTS mismatch")
        sample = index * SAMPLES_PER_FRAME
        timeline_matches = [
            entry
            for entry in timeline_payload
            if int(entry["output_start_sample"])
            <= sample
            < int(entry["output_end_sample_exclusive"])
        ]
        _require(len(timeline_matches) == 1, "frame sample does not map to exactly one timeline phase")
        expected_phase = timeline_matches[0]
        _require(row["phase"] == expected_phase["phase"], "frame map phase mismatch")
        active = [
            clip
            for clip in clips
            if int(clip["output_start_sample"])
            <= sample
            < int(clip["output_end_sample_exclusive"])
        ]
        _require(len(active) <= 1, "effective clip output intervals overlap")
        if active:
            clip = active[0]
            _require(row["phase"] == "clip", "active clip frame is not labeled clip")
            _require(row["clip_id"] == clip["clip_id"], "frame map clip mismatch")
            _require(row["group_id"] == clip["group_id"], "frame map group mismatch")
            expected_source_time = float(clip["source_start_s"]) + (
                sample - int(clip["output_start_sample"])
            ) / AUDIO_RATE
            _require(
                row.get("source_time_s", "") != ""
                and abs(float(row["source_time_s"]) - expected_source_time) <= 1e-8,
                "frame map source-time mismatch",
            )
        else:
            _require(
                not row["clip_id"]
                and not row["group_id"]
                and row.get("source_time_s", "") == "",
                "non-clip frame map row carries a clip/group/source time",
            )

    audio_meta, audio, audio_end = _decode_audio(video)
    _require(audio_meta == {"codec": "aac", "sample_rate": AUDIO_RATE, "channels": 2}, "audio must be AAC stereo 48 kHz")
    video_end = len(pts) / FPS
    _require(abs(audio_end - video_end) <= 0.06, "A/V end mismatch")
    audio_rows = _csv(audio_map)
    _require(len(audio_rows) == len(timeline_payload), "audio map/timeline row-count mismatch")
    previous = 0
    checks: list[dict[str, Any]] = []
    mapped_ids: list[str] = []
    for row_index, row in enumerate(audio_rows):
        expected_row = timeline_payload[row_index]
        _require(
            row["phase"] == str(expected_row["phase"])
            and row["clip_id"] == str(expected_row["clip_id"])
            and row["group_id"] == str(expected_row["group_id"]),
            "audio map phase/clip/group differs from effective timeline",
        )
        start, end = int(row["output_start_sample"]), int(row["output_end_sample_exclusive"])
        _require(
            start == int(expected_row["output_start_sample"])
            and end == int(expected_row["output_end_sample_exclusive"]),
            "audio map sample bounds differ from effective timeline",
        )
        _require(start == previous and end > start, "audio map gap, overlap, or empty phase")
        previous = end
        if row["phase"] != "clip":
            _require(
                not row["clip_id"]
                and not row["group_id"]
                and row.get("source_start_s", "") == ""
                and row.get("source_end_s", "") == "",
                "non-clip audio phase carries a clip/group/source interval",
            )
            _require(float(row["edge_fade_ms"]) == 0.0, "silence phase must not declare an edge fade")
            _require(abs(float(row["applied_global_gain"]) - float(payload["audio"]["applied_global_gain"])) <= 1e-12, "audio map global gain mismatch")
            continue
        clip = next(item for item in clips if item["clip_id"] == row["clip_id"])
        _require(row["group_id"] == clip["group_id"], "audio map group mismatch")
        _require(
            start == int(clip["output_start_sample"])
            and end == int(clip["output_end_sample_exclusive"]),
            "audio map clip bounds differ from effective clip bounds",
        )
        _require(
            abs(float(row["source_start_s"]) - float(clip["source_start_s"])) <= 1e-9
            and abs(float(row["source_end_s"]) - float(clip["source_end_s"])) <= 1e-9,
            "audio map source interval differs from effective clip interval",
        )
        _require(abs(float(row["edge_fade_ms"]) - float(payload["audio"]["edge_fade_ms"])) <= 1e-12, "audio map edge fade mismatch")
        _require(abs(float(row["applied_global_gain"]) - float(payload["audio"]["applied_global_gain"])) <= 1e-12, "audio map global gain mismatch")
        mapped_ids.append(clip["clip_id"])
        decoded = decode_audio(Path(clip["source"]), sample_rate=AUDIO_RATE, start_s=float(clip["source_start_s"]), end_s=float(clip["source_end_s"]))
        samples = decoded.samples
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        expected = end - start
        samples = np.pad(samples, ((0, max(0, expected - len(samples))), (0, 0)))[:expected]
        reference = apply_edge_fade(np.ascontiguousarray(samples.T, dtype=np.float32), float(payload["audio"]["edge_fade_ms"]))
        reference *= np.float32(payload["audio"]["applied_global_gain"])
        check = _preservation(reference, audio[:, start : min(end, audio.shape[1])])
        _require(check["normalized_correlation"] >= 0.97, f"{clip['clip_id']}: low source/output correlation")
        _require(abs(check["rms_delta_db"]) <= 1.5, f"{clip['clip_id']}: source/output RMS mismatch")
        _require(check["normalized_rmse"] <= 0.30, f"{clip['clip_id']}: source/output NRMSE too high")
        checks.append({"clip_id": clip["clip_id"], **check})
    _require(mapped_ids == [clip["clip_id"] for clip in clips], "audio map clip order mismatch")
    _require(previous == int(payload["audio"]["sample_count"]), "audio map terminal count mismatch")
    output_record = payload.get("output", {})
    _require(Path(str(output_record.get("video", ""))).resolve() == video and output_record.get("sha256") == sha256(video), "output video provenance mismatch")
    _require(Path(str(output_record.get("frame_map", ""))).resolve() == frame_map and output_record.get("frame_map_sha256") == sha256(frame_map), "frame map provenance mismatch")
    _require(Path(str(output_record.get("audio_map", ""))).resolve() == audio_map and output_record.get("audio_map_sha256") == sha256(audio_map), "audio map provenance mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    contact = output_dir / "VISUAL_QA_contact_sheet.png"
    _contact_sheet(video, clips, video_end, contact)
    result = {
        "status": "PASS",
        "scope": "descriptive signal rendering and A/V integrity only; no speaker identity",
        "hashes": {
            "video": sha256(video),
            "effective_manifest": sha256(effective),
            "frame_map": sha256(frame_map),
            "audio_map": sha256(audio_map),
            "source_manifest": payload["source_manifest_sha256"],
            "clip_manifest": clip_manifest["sha256"],
            "authoritative_acoustic_artifacts": {
                key: entry["sha256"] for key, entry in artifacts.items()
            },
            "layout_reference_asset": current_layout_reference["asset"]["sha256"],
            "layout_reference_manifest": current_layout_reference["manifest"]["sha256"],
            "approved_layout_preview": current_layout_review["preview"]["sha256"],
            "layout_preview_provenance": current_layout_review["preview_provenance"]["sha256"],
            "layout_review_sheet": current_layout_review["review_sheet"]["sha256"],
        },
        "effective_manifest_size_bytes": effective.stat().st_size,
        "video": {**video_meta, "frame_count": len(pts), "duration_s": video_end, "max_pts_error_s": max_pts_error, "maximum_limitation_strip_mae": max(banner_errors)},
        "audio": {**audio_meta, "end_s": audio_end, "av_end_difference_s": abs(audio_end - video_end), "clip_checks": checks},
        "mapping": {"frame_rows": len(frame_rows), "audio_phase_rows": len(audio_rows), "clip_order": mapped_ids},
        "visual_qa": {"status": "PENDING_FINAL_HUMAN_REVIEW", "approved_layout_preview": current_layout_review["preview"], "approved_layout_review_sheet": current_layout_review["review_sheet"], "contact_sheet": str(contact), "required_checks": ["final video remains conformant with the approved production preview and canonical composition reference", "designated anchor remains fixed in the left panel", "comparison group labels/colors and playheads map correctly", "zero lines, ranges, medians, intro/outro, and permanent limitation are readable"]},
        "prohibited_outputs": ["composite similarity", "distance", "ranking", "winner", "speaker identity", "nationality", "affiliation", "intent", "deception"],
    }
    (output_dir / "comparison_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "QA_RESULTS.md").write_text("# Voice-signal comparison QA\n\n**MACHINE PASS — FINAL HUMAN VISUAL/AUDITORY REVIEW PENDING**\n\n- Exact 24 fps CFR H.264/yuv420p at 1920×1080\n- AAC stereo 48 kHz\n- Canonical layout reference, approved production preview, side-by-side review sheet, and renderer-source provenance passed\n- Frame/audio maps, provenance hashes, source preservation, and permanent limitation passed\n- No composite, distance, rank, winner, or identity inference\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a designated-anchor voice-signal comparison.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--effective-manifest", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, required=True)
    parser.add_argument("--audio-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.video, args.effective_manifest, args.frame_map, args.audio_map, args.output_dir)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "comparison_verification.json").write_text(json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
