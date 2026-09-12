#!/usr/bin/env python3
"""Analyze bilabial closure presence and signed visible-release direction.

Audio release times come from the previously fixed audio-blinded annotations.
Mouth geometry comes from native-PTS ``lip_samples`` in the automated run.
Positive selected-edge lag means that the visible lip release follows audio.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

# Private aliases keep the historical script-level names importable for tests
# and callers that load this file as a module.
from video_integrity_analyzer.artifact_io import (  # noqa: E402
    atomic_write_text as _atomic_write_text,  # noqa: F401
    csv_cell as _csv_value,  # noqa: F401
    finite_float as _finite_float,
    json_ready as _json_ready,  # noqa: F401
    load_events_document,
    nested as _nested,
    now_utc as _now_utc,
    path_fingerprint as _path_fingerprint,
    visual_config_from_automated,
    write_csv as _write_csv,
    write_json as _write_json,
)
from video_integrity_analyzer.plosive_sync import (  # noqa: E402
    VisualReleaseConfig,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 1729
DEFAULT_CLOSURE_BEFORE_S = 0.250
DEFAULT_CLOSURE_AFTER_FRAMES = 2.0
DEFAULT_EDGE_WINDOW_S = 0.250
DEFAULT_MINIMUM_FACE_VALID_FRACTION = 0.60
DEFAULT_MAXIMUM_NEAREST_FRAME_ERROR_S = 0.050
DEFAULT_DIRECTION_MARGIN_FRAMES = 2.0


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    label: str
    phone_classes: tuple[str, ...]
    inferential_role: str


@dataclass(frozen=True)
class DirectionClosureProtocol:
    """Time and quality gates for one direction/closure analysis run."""

    fps: float
    cutoff_s: float | None
    closure_before_s: float
    closure_after_s: float
    edge_before_s: float
    edge_after_s: float
    minimum_face_valid_fraction: float
    maximum_nearest_frame_error_s: float
    direction_margin_frames: float

    @property
    def direction_margin_ms(self) -> float:
        return self.direction_margin_frames * 1000.0 / self.fps


def analysis_definitions() -> tuple[AnalysisSpec, ...]:
    return (
        AnalysisSpec("primary_p", "Primary: /p/", ("p",), "primary"),
        AnalysisSpec("secondary_b", "Secondary: /b/", ("b",), "secondary"),
    )


def _normalized_blinded_event(
    event: Mapping[str, object],
    automated_selection: Mapping[str, object],
) -> tuple[dict[str, object], str, float | None, object]:
    """Accept both the legacy nested form and the review finalizer schema.

    ``finalize_blinded_acoustic_review.py`` deliberately emits the reviewed
    fields at the event top level. Older callers and fixtures used a nested
    ``selection`` plus ``annotation_status``/``audio_release_time_s``. Keep
    both forms readable so the documented command chain composes directly.
    """

    nested_selection = _nested(event, "selection")
    if nested_selection:
        selection = dict(nested_selection)
    else:
        selection = dict(automated_selection)
        for key in ("speaker", "group", "epoch_id", "phoneme_class"):
            if event.get(key) is not None:
                selection[key] = event.get(key)

    status = str(event.get("annotation_status") or event.get("status") or "")
    audio_time = _finite_float(
        event.get("audio_release_time_s")
        if event.get("audio_release_time_s") is not None
        else event.get("selected_release_time_s")
    )
    confidence = _nested(event, "annotation").get("confidence")
    if confidence is None:
        confidence = event.get("confidence")
    return selection, status, audio_time, confidence


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    """Load an events document; only the canonical ``events`` key is accepted."""

    return load_events_document(path, label)


def _index_events(
    rows: Sequence[object], *, id_getter, label: str
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    duplicates: set[str] = set()
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} row {row_number} must be an object")
        event_id = str(id_getter(raw) or "").strip()
        if not event_id:
            raise ValueError(f"{label} row {row_number} lacks an event id")
        if event_id in indexed:
            duplicates.add(event_id)
        indexed[event_id] = raw
    if duplicates:
        raise ValueError(f"duplicate {label} event id: " + ", ".join(sorted(duplicates)))
    return indexed


def _normalized_samples(raw_samples: object) -> list[dict[str, object]]:
    if not isinstance(raw_samples, list):
        return []
    samples: list[dict[str, object]] = []
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            continue
        time_s = _finite_float(raw.get("time_s"))
        if time_s is None:
            continue
        samples.append(
            {
                "time_s": time_s,
                "median_aperture": _finite_float(raw.get("median_aperture")),
                "maximum_aperture": _finite_float(raw.get("maximum_aperture")),
                "aperture_spread": _finite_float(raw.get("aperture_spread")),
                "mouth_width_px": _finite_float(raw.get("mouth_width_px")),
                "face_quality": _finite_float(raw.get("face_quality")),
                "pair_apertures": raw.get("pair_apertures"),
                "repeated_frame": raw.get("repeated_frame") is True,
                "face_detected": raw.get("face_detected") is True,
                "source_pts": raw.get("source_pts"),
                "source_time_base": raw.get("source_time_base"),
            }
        )
    samples.sort(key=lambda item: float(item["time_s"]))
    if any(
        float(right["time_s"]) <= float(left["time_s"])
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("lip_samples timestamps must be unique and increasing")
    return samples


def classify_sample_quality(
    samples: Sequence[Mapping[str, object]], config: VisualReleaseConfig
) -> list[dict[str, object]]:
    """Attach the existing per-frame face gates and raw mouth-state labels."""

    normalized = [dict(sample) for sample in samples]
    width_jump = [False] * len(normalized)
    for index in range(1, len(normalized)):
        previous = _finite_float(normalized[index - 1].get("mouth_width_px"))
        current = _finite_float(normalized[index].get("mouth_width_px"))
        if previous is None or current is None or previous <= 0 or current <= 0:
            continue
        fraction = abs(current - previous) / max(current, previous, 1e-9)
        if fraction > config.maximum_mouth_width_jump_fraction:
            width_jump[index - 1] = True
            width_jump[index] = True

    classified: list[dict[str, object]] = []
    for index, sample in enumerate(normalized):
        median = _finite_float(sample.get("median_aperture"))
        maximum = _finite_float(sample.get("maximum_aperture"))
        spread = _finite_float(sample.get("aperture_spread"))
        width = _finite_float(sample.get("mouth_width_px"))
        quality = _finite_float(sample.get("face_quality"))
        reasons: list[str] = []
        if sample.get("face_detected") is not True:
            reasons.append("face_not_detected")
        if None in (median, maximum, spread, width, quality):
            reasons.append("nonfinite_mouth_geometry")
        if width is not None and width < config.minimum_mouth_width_px:
            reasons.append("mouth_resolution_too_low")
        if spread is not None and spread > config.maximum_pair_spread:
            reasons.append("inconsistent_lip_landmarks")
        if quality is not None and quality < config.minimum_face_quality:
            reasons.append("low_face_quality")
        if width_jump[index]:
            reasons.append("mouth_width_jump")
        quality_valid = not reasons
        usable = quality_valid and sample.get("repeated_frame") is not True
        contact = bool(
            usable
            and median is not None
            and maximum is not None
            and median <= config.closed_median_max
            and maximum <= config.closed_pair_max
        )
        opened = bool(
            usable
            and median is not None
            and median >= config.reopened_median_min
        )
        classified.append(
            {
                **sample,
                "quality_valid": quality_valid,
                "usable_nonrepeated": usable,
                "quality_exclusion_reasons": reasons,
                "contact": contact,
                "open": opened,
            }
        )
    return classified


def _samples_in_window(
    samples: Sequence[Mapping[str, object]], start_s: float, end_s: float
) -> list[Mapping[str, object]]:
    return [
        sample
        for sample in samples
        if start_s <= float(sample["time_s"]) <= end_s
    ]


def _edge_candidates(
    samples: Sequence[Mapping[str, object]],
    *,
    audio_time_s: float,
    config: VisualReleaseConfig,
    edge_before_s: float,
    edge_after_s: float,
) -> list[dict[str, object]]:
    window = _samples_in_window(
        samples, audio_time_s - edge_before_s, audio_time_s + edge_after_s
    )
    usable = [sample for sample in window if sample.get("usable_nonrepeated") is True]
    candidates: list[dict[str, object]] = []
    last_contact_index: int | None = None
    contact_run_count = 0
    for index, sample in enumerate(usable):
        time_s = float(sample["time_s"])
        if last_contact_index is not None:
            contact_time = float(usable[last_contact_index]["time_s"])
            if time_s - contact_time > config.maximum_release_gap_s:
                last_contact_index = None
                contact_run_count = 0
        if sample.get("contact") is True:
            if last_contact_index is not None and index == last_contact_index + 1:
                contact_run_count += 1
            else:
                contact_run_count = 1
            last_contact_index = index
            continue
        if sample.get("open") is not True or last_contact_index is None:
            continue
        contact = usable[last_contact_index]
        contact_time_s = float(contact["time_s"])
        gap_s = time_s - contact_time_s
        if 0.0 < gap_s <= config.maximum_release_gap_s:
            edge_time_s = (contact_time_s + time_s) * 0.5
            lag_ms = (edge_time_s - audio_time_s) * 1000.0
            candidates.append(
                {
                    "contact_frame_time_s": contact_time_s,
                    "open_frame_time_s": time_s,
                    "edge_time_s": edge_time_s,
                    "edge_lag_ms": lag_ms,
                    "contact_open_gap_ms": gap_s * 1000.0,
                    "contact_run_nonrepeated_frame_count": contact_run_count,
                    "contact_source_pts": contact.get("source_pts"),
                    "open_source_pts": sample.get("source_pts"),
                }
            )
        last_contact_index = None
        contact_run_count = 0
    return candidates


def classify_direction(lag_ms: float | None, *, margin_ms: float) -> str:
    if lag_ms is None:
        return "unavailable"
    if lag_ms < -margin_ms:
        return "video_leads"
    if lag_ms > margin_ms:
        return "video_lags"
    return "near"


def analyze_event_geometry(
    raw_samples: object,
    *,
    audio_time_s: float,
    fps: float,
    config: VisualReleaseConfig,
    protocol: DirectionClosureProtocol | None = None,
) -> dict[str, object]:
    if protocol is None:
        protocol = DirectionClosureProtocol(
            fps=fps,
            cutoff_s=None,
            closure_before_s=DEFAULT_CLOSURE_BEFORE_S,
            closure_after_s=DEFAULT_CLOSURE_AFTER_FRAMES / fps,
            edge_before_s=DEFAULT_EDGE_WINDOW_S,
            edge_after_s=DEFAULT_EDGE_WINDOW_S,
            minimum_face_valid_fraction=DEFAULT_MINIMUM_FACE_VALID_FRACTION,
            maximum_nearest_frame_error_s=DEFAULT_MAXIMUM_NEAREST_FRAME_ERROR_S,
            direction_margin_frames=DEFAULT_DIRECTION_MARGIN_FRAMES,
        )
    if not math.isclose(protocol.fps, fps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("protocol.fps must match fps")
    samples = classify_sample_quality(_normalized_samples(raw_samples), config)
    closure_window = _samples_in_window(
        samples,
        audio_time_s - protocol.closure_before_s,
        audio_time_s + protocol.closure_after_s,
    )
    usable = [
        sample
        for sample in closure_window
        if sample.get("usable_nonrepeated") is True
    ]
    valid_fraction = len(usable) / len(closure_window) if closure_window else 0.0
    nearest_error_ms = None
    if usable:
        nearest = min(usable, key=lambda item: abs(float(item["time_s"]) - audio_time_s))
        nearest_error_ms = (float(nearest["time_s"]) - audio_time_s) * 1000.0
    face_reasons: list[str] = []
    if not closure_window:
        face_reasons.append("no_native_pts_frames_in_closure_window")
    if valid_fraction < protocol.minimum_face_valid_fraction:
        face_reasons.append("insufficient_valid_nonrepeated_face_fraction")
    if (
        nearest_error_ms is None
        or abs(nearest_error_ms)
        > protocol.maximum_nearest_frame_error_s * 1000.0
    ):
        face_reasons.append("no_valid_nonrepeated_frame_near_audio_release")
    face_valid = not face_reasons

    contacts = [sample for sample in usable if sample.get("contact") is True]
    if not face_valid:
        closure_category = "unavailable"
    elif len(contacts) == 0:
        closure_category = "no_visible_contact"
    elif len(contacts) == 1:
        closure_category = "weak_contact"
    else:
        closure_category = "strong_contact"
    contact_offsets_ms = [
        (float(sample["time_s"]) - audio_time_s) * 1000.0 for sample in contacts
    ]
    one_frame_ms = 1000.0 / fps
    post_audio_only = bool(
        face_valid
        and contact_offsets_ms
        and min(contact_offsets_ms) > one_frame_ms
    )

    candidates = _edge_candidates(
        samples,
        audio_time_s=audio_time_s,
        config=config,
        edge_before_s=protocol.edge_before_s,
        edge_after_s=protocol.edge_after_s,
    ) if face_valid else []
    selected = (
        min(
            candidates,
            key=lambda item: (
                abs(float(item["edge_lag_ms"])),
                float(item["edge_time_s"]),
            ),
        )
        if candidates
        else None
    )
    margin_ms = protocol.direction_margin_ms
    selected_lag_ms = (
        float(selected["edge_lag_ms"]) if selected is not None else None
    )
    direction = classify_direction(selected_lag_ms, margin_ms=margin_ms)
    return {
        "face_valid": face_valid,
        "face_exclusion_reasons": face_reasons,
        "closure_window_start_s": audio_time_s - protocol.closure_before_s,
        "closure_window_end_s": audio_time_s + protocol.closure_after_s,
        "closure_window_native_frame_count": len(closure_window),
        "closure_window_usable_frame_count": len(usable),
        "closure_window_usable_fraction": valid_fraction,
        "nearest_usable_frame_error_ms": nearest_error_ms,
        "closure_category": closure_category,
        "contact_nonrepeated_frame_count": len(contacts),
        "contact_frame_times_s": [float(sample["time_s"]) for sample in contacts],
        "contact_frame_offsets_ms": contact_offsets_ms,
        "contact_source_pts": [sample.get("source_pts") for sample in contacts],
        "post_audio_only_contact": post_audio_only,
        "edge_search_start_s": audio_time_s - protocol.edge_before_s,
        "edge_search_end_s": audio_time_s + protocol.edge_after_s,
        "edge_candidates": candidates,
        "edge_candidate_count": len(candidates),
        "ambiguous_multiple_edges": len(candidates) > 1,
        "selected_edge": selected,
        "selected_edge_lag_ms": selected_lag_ms,
        "direction": direction,
    }


def build_records(
    blinded: Mapping[str, object],
    automated: Mapping[str, object],
    *,
    fps: float,
    config: VisualReleaseConfig,
    protocol: DirectionClosureProtocol | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if protocol is None:
        protocol = DirectionClosureProtocol(
            fps=fps,
            cutoff_s=None,
            closure_before_s=DEFAULT_CLOSURE_BEFORE_S,
            closure_after_s=DEFAULT_CLOSURE_AFTER_FRAMES / fps,
            edge_before_s=DEFAULT_EDGE_WINDOW_S,
            edge_after_s=DEFAULT_EDGE_WINDOW_S,
            minimum_face_valid_fraction=DEFAULT_MINIMUM_FACE_VALID_FRACTION,
            maximum_nearest_frame_error_s=DEFAULT_MAXIMUM_NEAREST_FRAME_ERROR_S,
            direction_margin_frames=DEFAULT_DIRECTION_MARGIN_FRAMES,
        )
    blind_rows = blinded.get("events")
    automated_rows = automated.get("events")
    assert isinstance(blind_rows, list) and isinstance(automated_rows, list)
    blind_index = _index_events(
        blind_rows,
        id_getter=lambda row: row.get("runner_event_id"),
        label="blinded",
    )
    auto_index = _index_events(
        automated_rows,
        id_getter=lambda row: _nested(row, "selection").get("event_id"),
        label="automated",
    )
    unknown = sorted(set(blind_index) - set(auto_index))
    if unknown:
        raise ValueError("blinded event ids missing from automated input: " + ", ".join(unknown))

    records: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    for event_id, blinded_event in blind_index.items():
        automated_event = auto_index[event_id]
        automated_selection = _nested(automated_event, "selection")
        selection, status, audio_time, confidence = _normalized_blinded_event(
            blinded_event, automated_selection
        )
        for key in ("speaker", "group", "epoch_id", "phoneme_class"):
            if selection.get(key) != automated_selection.get(key):
                raise ValueError(f"selection.{key} mismatch for {event_id}")
        token_start = _finite_float(selection.get("token_start_s"))
        target = True
        reasons: list[str] = []
        if status != "measurable" or audio_time is None:
            target = False
            reasons.append("audio_release_not_measurable")
        if protocol.cutoff_s is not None and (
            token_start is None or token_start >= protocol.cutoff_s
        ):
            target = False
            reasons.append("token_at_or_after_cutoff")
        if (
            protocol.cutoff_s is not None
            and audio_time is not None
            and audio_time >= protocol.cutoff_s
        ):
            target = False
            reasons.append("audio_release_at_or_after_cutoff")
        geometry = None
        if target and audio_time is not None:
            geometry = analyze_event_geometry(
                automated_event.get("lip_samples"),
                audio_time_s=audio_time,
                fps=fps,
                config=config,
                protocol=protocol,
            )
            if geometry["face_valid"] is not True:
                reasons.extend(str(reason) for reason in geometry["face_exclusion_reasons"])
        for reason in reasons:
            exclusions[reason] += 1
        records.append(
            {
                "runner_event_id": event_id,
                "selection": dict(selection),
                "audio_release_time_s": audio_time,
                "annotation_status": status,
                "annotation_confidence": confidence,
                "audio_measurable_in_scope": target,
                "analysis_eligible": bool(target and geometry and geometry["face_valid"] is True),
                "geometry": geometry,
                "exclusion_reasons": list(dict.fromkeys(reasons)),
            }
        )
    audit = {
        "status": "passed",
        "join_key": "blinded.runner_event_id == automated.selection.event_id",
        "blinded_event_count": len(blind_index),
        "automated_event_count": len(auto_index),
        "matched_event_count": len(blind_index),
        "audio_measurable_in_scope_count": sum(
            record["audio_measurable_in_scope"] is True for record in records
        ),
        "analysis_eligible_count": sum(
            record["analysis_eligible"] is True for record in records
        ),
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
        "unknown_blinded_event_count": 0,
        "duplicate_event_id_count": 0,
    }
    return records, audit


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _speaker_row(
    records: Sequence[Mapping[str, object]], spec: AnalysisSpec, speaker: str
) -> dict[str, object]:
    selected = [
        record
        for record in records
        if _nested(record, "selection").get("phoneme_class") in spec.phone_classes
        and _nested(record, "selection").get("speaker") == speaker
    ]
    eligible = [record for record in selected if record.get("analysis_eligible") is True]
    closures = Counter(_nested(record, "geometry").get("closure_category") for record in eligible)
    directions = Counter(_nested(record, "geometry").get("direction") for record in eligible)
    available_lags = [
        float(_nested(record, "geometry")["selected_edge_lag_ms"])
        for record in eligible
        if _finite_float(_nested(record, "geometry").get("selected_edge_lag_ms")) is not None
    ]
    denominator = len(eligible)
    group = str(_nested(selected[0], "selection").get("group")) if selected else "unknown"
    return {
        "analysis": spec.key,
        "inferential_role": spec.inferential_role,
        "speaker": speaker,
        "group": group,
        "selected_event_count": len(selected),
        "audio_measurable_in_scope_count": sum(record.get("audio_measurable_in_scope") is True for record in selected),
        "face_valid_event_count": denominator,
        "face_invalid_event_count": sum(
            record.get("audio_measurable_in_scope") is True and record.get("analysis_eligible") is not True
            for record in selected
        ),
        "epoch_count": len({_nested(record, "selection").get("epoch_id") for record in eligible}),
        "no_visible_contact_count": int(closures["no_visible_contact"]),
        "weak_contact_count": int(closures["weak_contact"]),
        "strong_contact_count": int(closures["strong_contact"]),
        "no_visible_contact_rate": closures["no_visible_contact"] / denominator if denominator else None,
        "not_strong_contact_rate": (closures["no_visible_contact"] + closures["weak_contact"]) / denominator if denominator else None,
        "post_audio_only_contact_count": sum(_nested(record, "geometry").get("post_audio_only_contact") is True for record in eligible),
        "video_leads_count": int(directions["video_leads"]),
        "near_count": int(directions["near"]),
        "video_lags_count": int(directions["video_lags"]),
        "direction_unavailable_count": int(directions["unavailable"]),
        "video_leads_rate": directions["video_leads"] / denominator if denominator else None,
        "video_lags_rate": directions["video_lags"] / denominator if denominator else None,
        "far_direction_rate": (directions["video_leads"] + directions["video_lags"]) / denominator if denominator else None,
        "direction_unavailable_rate": directions["unavailable"] / denominator if denominator else None,
        "multiple_edge_event_count": sum(_nested(record, "geometry").get("ambiguous_multiple_edges") is True for record in eligible),
        "median_selected_edge_lag_ms": _median(available_lags),
    }


def _epoch_rows(
    records: Sequence[Mapping[str, object]], spec: AnalysisSpec
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        selection = _nested(record, "selection")
        if (
            record.get("analysis_eligible") is True
            and selection.get("phoneme_class") in spec.phone_classes
        ):
            grouped[(str(selection.get("speaker")), str(selection.get("group")), str(selection.get("epoch_id")))].append(record)
    rows: list[dict[str, object]] = []
    for (speaker, group, epoch_id), events in sorted(grouped.items()):
        count = len(events)
        closures = Counter(_nested(event, "geometry").get("closure_category") for event in events)
        directions = Counter(_nested(event, "geometry").get("direction") for event in events)
        lags = [
            float(_nested(event, "geometry")["selected_edge_lag_ms"])
            for event in events
            if _finite_float(_nested(event, "geometry").get("selected_edge_lag_ms")) is not None
        ]
        rows.append(
            {
                "analysis": spec.key,
                "inferential_role": spec.inferential_role,
                "speaker": speaker,
                "group": group,
                "epoch_id": epoch_id,
                "event_count": count,
                "no_visible_contact_rate": closures["no_visible_contact"] / count,
                "not_strong_contact_rate": (closures["no_visible_contact"] + closures["weak_contact"]) / count,
                "post_audio_only_contact_rate": sum(_nested(event, "geometry").get("post_audio_only_contact") is True for event in events) / count,
                "video_leads_rate": directions["video_leads"] / count,
                "video_lags_rate": directions["video_lags"] / count,
                "far_direction_rate": (directions["video_leads"] + directions["video_lags"]) / count,
                "direction_unavailable_rate": directions["unavailable"] / count,
                "median_selected_edge_lag_ms": _median(lags),
                "available_edge_event_count": len(lags),
            }
        )
    return rows


def _pairwise_probability(candidate: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    differences = candidate[:, None] - control[None, :]
    wins = float(np.count_nonzero(differences > 0.0))
    ties = float(np.count_nonzero(differences == 0.0))
    probability = (wins + 0.5 * ties) / float(differences.size)
    return probability, 2.0 * probability - 1.0


def _bootstrap_ci(
    candidate: np.ndarray, control: np.ndarray, *, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    candidate_draws = candidate[
        rng.integers(0, candidate.size, size=(iterations, candidate.size))
    ]
    control_draws = control[
        rng.integers(0, control.size, size=(iterations, control.size))
    ]
    differences = np.median(candidate_draws, axis=1) - np.median(control_draws, axis=1)
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def _central_index_options(population_size: int, group_size: int):
    if group_size % 2:
        return ((index,) for index in range(population_size))
    return itertools.combinations(range(population_size), 2)


def _median_orders(group_size: int) -> tuple[int, ...]:
    if group_size % 2:
        return ((group_size + 1) // 2,)
    return (group_size // 2, group_size // 2 + 1)


def _labeling_count(
    population_size: int,
    candidate_size: int,
    candidate_indices: Sequence[int],
    control_indices: Sequence[int],
) -> int:
    constraints: dict[int, tuple[int, int]] = {}

    def add(index: int, status: int, cumulative: int) -> bool:
        value = (status, cumulative)
        previous = constraints.get(index)
        if previous is not None and previous != value:
            return False
        constraints[index] = value
        return True

    for index, order in zip(candidate_indices, _median_orders(candidate_size), strict=True):
        if not add(index, 1, order):
            return 0
    control_size = population_size - candidate_size
    for index, order in zip(control_indices, _median_orders(control_size), strict=True):
        if not add(index, 0, (index + 1) - order):
            return 0
    ways = 1
    previous_index = -1
    previous_cumulative = 0
    for index, (status, cumulative) in sorted(constraints.items()):
        gap = index - previous_index - 1
        selected = cumulative - previous_cumulative - status
        if selected < 0 or selected > gap:
            return 0
        ways *= math.comb(gap, selected)
        previous_index = index
        previous_cumulative = cumulative
    tail = population_size - previous_index - 1
    selected_tail = candidate_size - previous_cumulative
    if selected_tail < 0 or selected_tail > tail:
        return 0
    return ways * math.comb(tail, selected_tail)


def exact_one_sided_median_permutation(
    candidate: np.ndarray, control: np.ndarray
) -> dict[str, object]:
    """Exact candidate-greater label permutation for a median difference."""

    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if candidate.size == 0 or control.size == 0:
        return {"p_value": None, "tail_labeling_count": 0, "total_labeling_count": 0}
    combined = np.sort(np.concatenate((candidate, control)), kind="stable")
    population_size = int(combined.size)
    candidate_size = int(candidate.size)
    observed = float(np.median(candidate) - np.median(control))
    tolerance = 1e-12 * max(1.0, abs(observed))
    tail_count = 0
    counted = 0
    for candidate_indices in _central_index_options(population_size, candidate_size):
        candidate_median = float(np.mean(combined[list(candidate_indices)]))
        for control_indices in _central_index_options(population_size, population_size - candidate_size):
            ways = _labeling_count(
                population_size, candidate_size, candidate_indices, control_indices
            )
            if not ways:
                continue
            statistic = candidate_median - float(np.mean(combined[list(control_indices)]))
            counted += ways
            if statistic >= observed - tolerance:
                tail_count += ways
    expected = math.comb(population_size, candidate_size)
    if counted != expected:
        raise RuntimeError(f"exact permutation audit failed: {counted} != {expected}")
    return {
        "p_value": float(tail_count / expected),
        "tail_labeling_count": int(tail_count),
        "total_labeling_count": int(expected),
    }


def compare_epoch_metric(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    candidate = np.asarray(
        [float(row[metric]) for row in rows if row.get("group") == "candidate" and _finite_float(row.get(metric)) is not None],
        dtype=np.float64,
    )
    control = np.asarray(
        [float(row[metric]) for row in rows if row.get("group") == "control" and _finite_float(row.get(metric)) is not None],
        dtype=np.float64,
    )
    result = {
        "metric": metric,
        "alternative": "candidate_greater_than_pooled_controls",
        "comparison_unit": "epoch",
        "candidate_epoch_count": int(candidate.size),
        "control_epoch_count": int(control.size),
        "candidate_median": _median(candidate.tolist()),
        "control_median": _median(control.tolist()),
        "median_difference_candidate_minus_control": None,
        "cluster_bootstrap_ci95": None,
        "cluster_bootstrap_iterations": bootstrap_iterations,
        "probability_candidate_greater": None,
        "cliffs_delta": None,
        "exploratory_one_sided_exact_permutation_p": None,
        "exact_permutation_tail_labeling_count": 0,
        "exact_permutation_total_labeling_count": 0,
        "status": "inconclusive_missing_group",
    }
    if candidate.size == 0 or control.size == 0:
        return result
    probability, delta = _pairwise_probability(candidate, control)
    exact = exact_one_sided_median_permutation(candidate, control)
    result.update(
        median_difference_candidate_minus_control=float(np.median(candidate) - np.median(control)),
        cluster_bootstrap_ci95=_bootstrap_ci(candidate, control, iterations=bootstrap_iterations, seed=seed),
        probability_candidate_greater=probability,
        cliffs_delta=delta,
        exploratory_one_sided_exact_permutation_p=exact["p_value"],
        exact_permutation_tail_labeling_count=exact["tail_labeling_count"],
        exact_permutation_total_labeling_count=exact["total_labeling_count"],
        status="descriptive_with_exploratory_inference",
    )
    return result


METRICS = (
    "no_visible_contact_rate",
    "not_strong_contact_rate",
    "post_audio_only_contact_rate",
    "video_leads_rate",
    "video_lags_rate",
    "far_direction_rate",
    "direction_unavailable_rate",
    "median_selected_edge_lag_ms",
)


def build_analyses(
    records: Sequence[Mapping[str, object]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    speakers = sorted(
        {
            str(_nested(record, "selection").get("speaker"))
            for record in records
            if _nested(record, "selection").get("speaker") is not None
        },
        key=lambda value: value,
    )
    analyses: dict[str, object] = {}
    speaker_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for analysis_index, spec in enumerate(analysis_definitions()):
        current_speakers = [_speaker_row(records, spec, speaker) for speaker in speakers]
        current_epochs = _epoch_rows(records, spec)
        comparisons: dict[str, object] = {}
        for metric_index, metric in enumerate(METRICS):
            comparison = compare_epoch_metric(
                current_epochs,
                metric=metric,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + analysis_index * 100 + metric_index,
            )
            comparisons[metric] = comparison
            comparison_rows.append(
                {
                    "analysis": spec.key,
                    "inferential_role": spec.inferential_role,
                    **comparison,
                }
            )
        eligible = [
            record
            for record in records
            if record.get("analysis_eligible") is True
            and _nested(record, "selection").get("phoneme_class") in spec.phone_classes
        ]
        analyses[spec.key] = {
            "definition": asdict(spec),
            "counts": {
                "analysis_eligible_event_count": len(eligible),
                "eligible_epoch_count": len(current_epochs),
                "multiple_edge_event_count": sum(_nested(record, "geometry").get("ambiguous_multiple_edges") is True for record in eligible),
            },
            "speaker_event_descriptives": current_speakers,
            "epoch_descriptives": current_epochs,
            "candidate_vs_pooled_controls": comparisons,
        }
        speaker_rows.extend(current_speakers)
        epoch_rows.extend(current_epochs)
    return analyses, speaker_rows, epoch_rows, comparison_rows


def event_csv_row(record: Mapping[str, object]) -> dict[str, object]:
    selection = _nested(record, "selection")
    geometry = _nested(record, "geometry")
    selected_edge = _nested(geometry, "selected_edge")
    return {
        "runner_event_id": record.get("runner_event_id"),
        "speaker": selection.get("speaker"),
        "group": selection.get("group"),
        "epoch_id": selection.get("epoch_id"),
        "phoneme_class": selection.get("phoneme_class"),
        "token_start_s": selection.get("token_start_s"),
        "audio_release_time_s": record.get("audio_release_time_s"),
        "annotation_status": record.get("annotation_status"),
        "audio_measurable_in_scope": record.get("audio_measurable_in_scope"),
        "analysis_eligible": record.get("analysis_eligible"),
        "closure_category": geometry.get("closure_category"),
        "contact_nonrepeated_frame_count": geometry.get("contact_nonrepeated_frame_count"),
        "contact_frame_times_s": geometry.get("contact_frame_times_s"),
        "contact_frame_offsets_ms": geometry.get("contact_frame_offsets_ms"),
        "post_audio_only_contact": geometry.get("post_audio_only_contact"),
        "closure_window_native_frame_count": geometry.get("closure_window_native_frame_count"),
        "closure_window_usable_frame_count": geometry.get("closure_window_usable_frame_count"),
        "closure_window_usable_fraction": geometry.get("closure_window_usable_fraction"),
        "edge_candidate_count": geometry.get("edge_candidate_count"),
        "ambiguous_multiple_edges": geometry.get("ambiguous_multiple_edges"),
        "direction": geometry.get("direction"),
        "selected_edge_time_s": selected_edge.get("edge_time_s"),
        "selected_edge_lag_ms": geometry.get("selected_edge_lag_ms"),
        "all_edge_candidates": geometry.get("edge_candidates"),
        "exclusion_reasons": record.get("exclusion_reasons"),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    blinded_path = args.blinded_events_json.expanduser().resolve()
    automated_path = args.automated_events_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (blinded_path, automated_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = (
        "analysis.json",
        "events.json",
        "events.csv",
        "speaker_events.csv",
        "epochs.csv",
        "comparisons.csv",
    )
    existing = [name for name in planned if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("output files already exist; pass --overwrite: " + ", ".join(existing))

    blinded = _load_json_object(blinded_path, "blinded events JSON")
    automated = _load_json_object(automated_path, "automated events JSON")
    fps = _finite_float(getattr(args, "fps", None))
    if fps is None:
        fps = _finite_float(_nested(automated, "media").get("average_fps"))
    if fps is None or fps <= 0:
        raise ValueError("fps override or automated media.average_fps must be positive")
    raw_cutoff_s = getattr(args, "cutoff_s", None)
    cutoff_s = _finite_float(raw_cutoff_s)
    if raw_cutoff_s is not None and cutoff_s is None:
        raise ValueError("cutoff_s must be finite when supplied")
    closure_before_s = float(
        getattr(args, "closure_before_s", DEFAULT_CLOSURE_BEFORE_S)
    )
    closure_after_arg = getattr(args, "closure_after_s", None)
    closure_after_s = (
        DEFAULT_CLOSURE_AFTER_FRAMES / fps
        if closure_after_arg is None
        else float(closure_after_arg)
    )
    edge_before_s = float(
        getattr(args, "edge_before_s", DEFAULT_EDGE_WINDOW_S)
    )
    edge_after_s = float(getattr(args, "edge_after_s", DEFAULT_EDGE_WINDOW_S))
    minimum_face_valid_fraction = float(
        getattr(
            args,
            "minimum_face_valid_fraction",
            DEFAULT_MINIMUM_FACE_VALID_FRACTION,
        )
    )
    maximum_nearest_frame_error_s = float(
        getattr(
            args,
            "maximum_nearest_frame_error_s",
            DEFAULT_MAXIMUM_NEAREST_FRAME_ERROR_S,
        )
    )
    direction_margin_frames = float(
        getattr(args, "direction_margin_frames", DEFAULT_DIRECTION_MARGIN_FRAMES)
    )
    protocol_numbers = (
        closure_before_s,
        closure_after_s,
        edge_before_s,
        edge_after_s,
        minimum_face_valid_fraction,
        maximum_nearest_frame_error_s,
        direction_margin_frames,
    )
    if not all(math.isfinite(value) for value in protocol_numbers):
        raise ValueError("protocol values must be finite")
    if cutoff_s is not None and cutoff_s <= 0:
        raise ValueError("cutoff_s must be positive when supplied")
    if min(
        closure_before_s,
        closure_after_s,
        edge_before_s,
        edge_after_s,
        maximum_nearest_frame_error_s,
        direction_margin_frames,
    ) <= 0:
        raise ValueError(
            "window, frame-error, and direction-margin values must be positive"
        )
    if not 0 < minimum_face_valid_fraction <= 1:
        raise ValueError("minimum_face_valid_fraction must be in (0, 1]")
    protocol = DirectionClosureProtocol(
        fps=fps,
        cutoff_s=cutoff_s,
        closure_before_s=closure_before_s,
        closure_after_s=closure_after_s,
        edge_before_s=edge_before_s,
        edge_after_s=edge_after_s,
        minimum_face_valid_fraction=minimum_face_valid_fraction,
        maximum_nearest_frame_error_s=maximum_nearest_frame_error_s,
        direction_margin_frames=direction_margin_frames,
    )
    config = visual_config_from_automated(automated)
    records, audit = build_records(
        blinded, automated, fps=fps, config=config, protocol=protocol
    )
    analyses, speaker_rows, epoch_rows, comparison_rows = build_analyses(
        records,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    created_utc = _now_utc()
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "status": "direction_and_visible_closure_analysis",
        "inputs": {
            "blinded_events_json": _path_fingerprint(blinded_path),
            "automated_events_json": _path_fingerprint(automated_path),
        },
        "configuration": {
            "cutoff_s_exclusive": protocol.cutoff_s,
            "fps": fps,
            "direction_margin_frames": protocol.direction_margin_frames,
            "direction_margin_ms": protocol.direction_margin_ms,
            "closure_search_before_ms": protocol.closure_before_s * 1000.0,
            "closure_search_after_ms": protocol.closure_after_s * 1000.0,
            "edge_search_before_ms": protocol.edge_before_s * 1000.0,
            "edge_search_after_ms": protocol.edge_after_s * 1000.0,
            "minimum_face_valid_nonrepeated_fraction": protocol.minimum_face_valid_fraction,
            "maximum_nearest_frame_error_ms": protocol.maximum_nearest_frame_error_s
            * 1000.0,
            "visual_release_config": asdict(config),
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
        "join_and_eligibility_audit": audit,
        "analyses": analyses,
        "selection_bias": (
            "When multiple contact-to-open edges exist, the edge nearest the fixed "
            "audio release is selected without using speaker labels. This can bias "
            "absolute lag toward zero; every candidate edge is retained in events.json."
        ),
        "limitations": [
            "No-visible-contact means no usable non-repeated contact frame was detected; it does not prove that the lips never contacted.",
            "Either lag direction can arise from capture, encoding, network, conferencing, or playback synchronization.",
            "Participant stream is confounded with device, network path, camera, microphone, and Meet buffering.",
            "Exact permutation p-values are exploratory because speaker streams and epochs were not randomized or exchangeable.",
            "The analysis cannot identify lip-syncing, a proxy speaker, identity, nationality, or affiliation.",
        ],
    }
    _write_json(output_dir / "analysis.json", analysis)
    _write_json(
        output_dir / "events.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "join_and_eligibility_audit": audit,
            "events": records,
        },
    )
    _write_csv(output_dir / "events.csv", [event_csv_row(record) for record in records])
    _write_csv(output_dir / "speaker_events.csv", speaker_rows)
    _write_csv(output_dir / "epochs.csv", epoch_rows)
    _write_csv(output_dir / "comparisons.csv", comparison_rows)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blinded-events-json", required=True, type=Path)
    parser.add_argument("--automated-events-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cutoff-s",
        type=float,
        help="Optional exclusive analysis cutoff in source seconds; default: no cutoff.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Override media.average_fps for frame-derived thresholds.",
    )
    parser.add_argument(
        "--closure-before-s", type=float, default=DEFAULT_CLOSURE_BEFORE_S
    )
    parser.add_argument(
        "--closure-after-s",
        type=float,
        help="Closure window after release; default: two frames at effective FPS.",
    )
    parser.add_argument(
        "--edge-before-s", type=float, default=DEFAULT_EDGE_WINDOW_S
    )
    parser.add_argument(
        "--edge-after-s", type=float, default=DEFAULT_EDGE_WINDOW_S
    )
    parser.add_argument(
        "--minimum-face-valid-fraction",
        type=float,
        default=DEFAULT_MINIMUM_FACE_VALID_FRACTION,
    )
    parser.add_argument(
        "--maximum-nearest-frame-error-s",
        type=float,
        default=DEFAULT_MAXIMUM_NEAREST_FRAME_ERROR_S,
    )
    parser.add_argument(
        "--direction-margin-frames",
        type=float,
        default=DEFAULT_DIRECTION_MARGIN_FRAMES,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise SystemExit("--bootstrap-iterations must be positive")
    analysis = run(args)
    audit = analysis["join_and_eligibility_audit"]
    print(
        "Direction/closure analysis complete: "
        f"{audit['audio_measurable_in_scope_count']} audio-measurable in scope, "
        f"{audit['analysis_eligible_count']} face-valid events."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
