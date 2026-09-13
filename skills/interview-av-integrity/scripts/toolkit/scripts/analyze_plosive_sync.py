#!/usr/bin/env python3
"""Measure bilabial audio/video release timing in a recorded interview.

ASR is used only to preselect Japanese /p/ and /b/ kana and to provide a
search anchor.  The acoustic release is refined from the waveform while the
visible release is measured independently from native video PTS and FaceMesh
lip geometry.  Positive lag means that the visible release followed the
acoustic release.

The output is an auditable measurement bundle, not a deepfake or identity
classifier.  Participant, device, network, conferencing, and encoder effects
remain confounded in a single call recording.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

# Private aliases keep the historical script-level names importable for tests
# and callers that load this file as a module.
from video_integrity_analyzer.artifact_io import (  # noqa: E402
    atomic_write_text as _atomic_write_text,  # noqa: F401
    csv_cell as _json_cell,  # noqa: F401
    json_ready as _json_ready,  # noqa: F401
    nested as _nested,
    now_utc as _now_utc,
    path_fingerprint as _path_fingerprint,
    write_csv,
    write_json as _write_json,
)
from video_integrity_analyzer.media import probe_media, probe_video_frame_timing  # noqa: E402
from video_integrity_analyzer.face_process import preflight_face_runtime  # noqa: E402
from video_integrity_analyzer.plosive_manifest import (  # noqa: E402
    BilabialToken,
    extract_bilabial_tokens,
    load_speaker_intervals,
)
from video_integrity_analyzer.plosive_stats import (  # noqa: E402
    compare_candidate_to_controls,
    summarize_epochs,
)
from video_integrity_analyzer.plosive_sync import (  # noqa: E402
    AcousticReleaseConfig,
    VisualReleaseConfig,
    combine_release_estimates,
    decode_audio_window,
    estimate_acoustic_release,
    estimate_visual_release,
    extract_native_lip_samples,
    identify_mouth_shape_at_acoustic_release,
    lip_samples_as_records,
)


SCHEMA_VERSION = 1
MINIMUM_ASR_PROBABILITY = 0.75
CONSERVATIVE_MINIMUM_ASR_PROBABILITY = 0.80
BOUNDARY_GUARD_S = 0.75
AUDIO_SAMPLE_RATE = 48_000
EVENT_WINDOW_BEFORE_S = 0.75
EVENT_WINDOW_AFTER_S = 0.60
DEFAULT_SEED = 1729
SUPPORTED_MEDIA_PROFILE = {
    "video_codec": "h264",
    "width": 1920,
    "height": 1080,
    "average_fps": 24.0,
    "audio_codec": "aac",
    "audio_sample_rate": 48_000,
}


def _validate_supported_media_profile(media: Any) -> None:
    """Fail closed outside the documented, uncalibrated reference profile."""

    mismatches: list[str] = []
    for field in ("video_codec", "width", "height", "audio_codec", "audio_sample_rate"):
        expected = SUPPORTED_MEDIA_PROFILE[field]
        actual = getattr(media, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")
    fps = float(getattr(media, "average_fps", math.nan))
    if not math.isfinite(fps) or abs(fps - 24.0) > 0.01:
        mismatches.append(f"average_fps={fps!r} (expected 24.0)")
    if mismatches:
        raise ValueError(
            "fixed acoustic/visual gates are limited to the documented Meet-style "
            "H.264 1920x1080 24fps/AAC48k profile; calibrate a separate method "
            "instead of reusing them: " + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    directory: str
    label: str
    phone_classes: tuple[str, ...]
    inferential_role: str


def analysis_definitions() -> tuple[AnalysisSpec, ...]:
    """Return the predeclared primary, secondary, and exploratory strata."""

    return (
        AnalysisSpec(
            key="primary_p",
            directory="primary_p",
            label="Primary: clear /p/ releases",
            phone_classes=("p",),
            inferential_role="primary",
        ),
        AnalysisSpec(
            key="secondary_b",
            directory="secondary_b",
            label="Secondary: /b/ releases (lenition-sensitive)",
            phone_classes=("b",),
            inferential_role="secondary",
        ),
        AnalysisSpec(
            key="combined_exploratory",
            directory="combined_exploratory",
            label="Exploratory: pooled /p/ and /b/ releases",
            phone_classes=("p", "b"),
            inferential_role="exploratory",
        ),
    )


def enforce_speaker_token_manifest(
    tokens: Sequence[BilabialToken],
    manifest: Mapping[str, object],
) -> tuple[list[BilabialToken], dict[str, dict[str, object]], dict[str, object]]:
    """Replace approximate interval labels with token-level guarded labels.

    The finalized speaker-token manifest samples the on-screen name label at the
    token and at +/-0.75 s. It therefore supersedes the 2 Hz compressed
    interval boundary approximation used to create the initial token list.
    Matching uses the original ASR word start/end, target kana, and occurrence
    within that word; no face or voice identity feature enters the join.
    """

    raw_rows = manifest.get("tokens")
    if not isinstance(raw_rows, list):
        raise ValueError("speaker-token manifest must contain a tokens list")

    method = manifest.get("method")
    reference_labels = (
        method.get("reference_labels") if isinstance(method, Mapping) else None
    )
    configured_groups: dict[str, str] = {}
    if isinstance(reference_labels, Mapping):
        for speaker_id, reference in reference_labels.items():
            if isinstance(reference, Mapping) and reference.get("group") is not None:
                configured_groups[str(speaker_id)] = str(reference["group"])

    def key_for_row(row: Mapping[str, object]) -> tuple[float, float, str, int]:
        return (
            round(float(row["asr_word_start_s"]), 6),
            round(float(row["asr_word_end_s"]), 6),
            str(row["kana"]),
            int(row["occurrence_in_word"]),
        )

    indexed: dict[tuple[float, float, str, int], Mapping[str, object]] = {}
    duplicate_keys: list[tuple[float, float, str, int]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("speaker-token manifest rows must be objects")
        key = key_for_row(raw)
        if key in indexed:
            duplicate_keys.append(key)
        indexed[key] = raw
    if duplicate_keys:
        raise ValueError(
            f"speaker-token manifest has {len(duplicate_keys)} duplicate join keys"
        )

    occurrence_by_word: Counter[tuple[float, float]] = Counter()
    joined_tokens: list[BilabialToken] = []
    metadata_by_event: dict[str, dict[str, object]] = {}
    used_keys: set[tuple[float, float, str, int]] = set()
    counts: Counter[str] = Counter()
    for token in tokens:
        word_key = (round(token.token_start_s, 6), round(token.token_end_s, 6))
        occurrence = occurrence_by_word[word_key]
        occurrence_by_word[word_key] += 1
        key = (*word_key, token.kana, occurrence)
        row = indexed.get(key)
        interval_reasons = (
            token.exclusion_reason.split(";") if token.exclusion_reason else []
        )
        # Direct token-level guard observations supersede these two reasons.
        retained_reasons = [
            reason
            for reason in interval_reasons
            if reason
            not in {
                "near_active_speaker_transition",
                "outside_labeled_speaker_interval",
            }
        ]
        metadata: dict[str, object] = {
            "join_key": {
                "asr_word_start_s": word_key[0],
                "asr_word_end_s": word_key[1],
                "kana": token.kana,
                "occurrence_in_word": occurrence,
            },
            "interval_speaker": token.speaker,
            "interval_group": token.group,
            "interval_exclusion_reason": token.exclusion_reason,
            "interval_boundary_reason_superseded": len(retained_reasons)
            != len(interval_reasons),
        }
        if row is None:
            counts["missing_manifest_match"] += 1
            retained_reasons.append("speaker_token_manifest_missing")
            metadata.update(
                manifest_token_id=None,
                manifest_speaker=None,
                speaker_stable_for_analysis=False,
                speaker_label_mismatch=False,
                matched=False,
            )
            joined = replace(
                token,
                eligible=False,
                exclusion_reason=";".join(dict.fromkeys(retained_reasons)),
            )
        else:
            used_keys.add(key)
            counts["matched"] += 1
            manifest_speaker = str(row.get("speaker_id", "unknown"))
            row_group = row.get("group")
            configured_group = configured_groups.get(manifest_speaker)
            manifest_group = (
                str(row_group)
                if row_group is not None
                else configured_group
                if configured_group is not None
                else token.group
            )
            group_source = (
                "token_row"
                if row_group is not None
                else "reference_config"
                if configured_group is not None
                else "speaker_interval_fallback"
            )
            group_configuration_mismatch = (
                row_group is not None
                and configured_group is not None
                and str(row_group) != configured_group
            )
            label_guard_stable = bool(row.get("speaker_stable_for_analysis"))
            analysis_group = manifest_group in {"candidate", "control"}
            stable = (
                label_guard_stable
                and manifest_speaker != "unknown"
                and analysis_group
                and not group_configuration_mismatch
            )
            mismatch = token.speaker is not None and token.speaker != manifest_speaker
            group_mismatch = (
                token.group is not None
                and manifest_group is not None
                and token.group != manifest_group
            )
            counts["stable"] += int(stable)
            counts["unstable"] += int(not stable)
            counts["speaker_label_mismatch"] += int(mismatch)
            counts["group_label_mismatch"] += int(group_mismatch)
            counts["group_configuration_mismatch"] += int(
                group_configuration_mismatch
            )
            if not label_guard_stable or manifest_speaker == "unknown":
                retained_reasons.append("speaker_token_manifest_unstable")
            if not analysis_group:
                retained_reasons.append("speaker_token_manifest_nonanalysis_group")
            if group_configuration_mismatch:
                retained_reasons.append("speaker_token_manifest_group_mismatch")
            metadata.update(
                manifest_token_id=row.get("token_id"),
                manifest_speaker=manifest_speaker,
                manifest_group=manifest_group,
                manifest_group_source=group_source,
                configured_group=configured_group,
                group_configuration_mismatch=group_configuration_mismatch,
                interval_group_mismatch=group_mismatch,
                label_guard_stable=label_guard_stable,
                speaker_stable_for_analysis=stable,
                speaker_label_mismatch=mismatch,
                matched=True,
            )
            joined = replace(
                token,
                speaker=manifest_speaker if manifest_speaker != "unknown" else None,
                group=manifest_group,
                eligible=not retained_reasons,
                exclusion_reason=(
                    ";".join(dict.fromkeys(retained_reasons))
                    if retained_reasons
                    else None
                ),
            )
        counts["eligible_before_join"] += int(token.eligible)
        counts["eligible_after_join"] += int(joined.eligible)
        metadata["final_eligible"] = joined.eligible
        metadata["final_exclusion_reason"] = joined.exclusion_reason
        joined_tokens.append(joined)
        metadata_by_event[token.event_id] = metadata

    counts["tokens"] = len(tokens)
    counts["manifest_rows"] = len(raw_rows)
    counts["unused_manifest_rows"] = len(set(indexed) - used_keys)
    audit = {
        "method": (
            "exact join on ASR word start/end, target kana, and occurrence; "
            "token-level display-name guard supersedes compressed interval boundaries"
        ),
        "counts": dict(counts),
        "events": metadata_by_event,
    }
    return joined_tokens, metadata_by_event, audit


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a table atomically; an empty table keeps its ``empty`` placeholder column."""

    write_csv(path, rows, empty_placeholder_column="empty")


def event_csv_row(record: Mapping[str, object]) -> dict[str, object]:
    """Flatten one audit event while keeping arrays JSON-encoded in CSV."""

    selection = _nested(record, "selection")
    measurement = _nested(record, "measurement")
    acoustic = _nested(measurement, "acoustic")
    acoustic_candidate = _nested(acoustic, "selected_candidate")
    visual = _nested(measurement, "visual")
    shape = _nested(measurement, "mouth_shape_at_acoustic_release")
    diagnostic_shape = _nested(record, "diagnostic_mouth_shape_at_top_acoustic_candidate")
    speaker_join = _nested(record, "speaker_token_manifest_join")
    audio_window = _nested(record, "audio_window")
    lip_samples = record.get("lip_samples")
    lip_sample_count = len(lip_samples) if isinstance(lip_samples, list) else 0
    return {
        "event_id": selection.get("event_id"),
        "speaker": selection.get("speaker"),
        "group": selection.get("group"),
        "epoch_id": selection.get("epoch_id"),
        "phoneme_class": selection.get("phoneme_class"),
        "kana": selection.get("kana"),
        "token_text": selection.get("token_text"),
        "reading_text": selection.get("reading_text"),
        "reading_source": selection.get("reading_source"),
        "token_start_s": selection.get("token_start_s"),
        "token_end_s": selection.get("token_end_s"),
        "asr_anchor_s": selection.get("anchor_s"),
        "asr_probability": selection.get("asr_probability"),
        "selection_eligible": selection.get("eligible"),
        "selection_exclusion_reason": selection.get("exclusion_reason"),
        "speaker_manifest_token_id": speaker_join.get("manifest_token_id"),
        "speaker_manifest_matched": speaker_join.get("matched"),
        "speaker_manifest_stable": speaker_join.get("speaker_stable_for_analysis"),
        "speaker_manifest_label_mismatch": speaker_join.get(
            "speaker_label_mismatch"
        ),
        "speaker_manifest_boundary_reason_superseded": speaker_join.get(
            "interval_boundary_reason_superseded"
        ),
        "attempted": record.get("attempted"),
        "status": record.get("status"),
        "measurable": record.get("measurable"),
        "release_lag_ms": record.get("release_lag_ms"),
        "diagnostic_release_lag_ms": record.get("diagnostic_release_lag_ms"),
        "diagnostic_candidate_lag_ms": record.get("diagnostic_candidate_lag_ms"),
        "acoustic_auto_accepted": record.get("acoustic_auto_accepted"),
        "acoustic_auto_acceptance_reasons": record.get(
            "acoustic_auto_acceptance_reasons"
        ),
        "core_combined_measurable": measurement.get("measurable"),
        "combined_timing_uncertainty_ms": measurement.get(
            "combined_timing_uncertainty_ms"
        ),
        "event_exclusion_reasons": record.get("exclusion_reasons"),
        "audio_window_start_s": audio_window.get("start_s"),
        "audio_window_end_s": audio_window.get("end_s"),
        "audio_coverage_fraction": audio_window.get("coverage_fraction"),
        "audio_warnings": audio_window.get("warnings"),
        "acoustic_candidate_time_s": acoustic.get("candidate_time_s"),
        "acoustic_release_time_s": acoustic.get("release_time_s"),
        "acoustic_score": acoustic.get("score"),
        "acoustic_runner_up_margin": acoustic.get("runner_up_margin"),
        "acoustic_acceptance_mode": acoustic.get("acceptance_mode"),
        "acoustic_confidence": acoustic.get("confidence"),
        "acoustic_measurable": acoustic.get("measurable"),
        "acoustic_phoneme_identity_status": acoustic.get("phoneme_identity_status"),
        "acoustic_exclusion_reasons": acoustic.get("exclusion_reasons"),
        # Retain evidence for rejected peaks even when no marker is selected.
        "acoustic_candidates": acoustic.get("candidates"),
        "acoustic_broadband_release_evidence": acoustic_candidate.get(
            "broadband_release_evidence"
        ),
        "acoustic_burst_spectral_flatness": acoustic_candidate.get(
            "burst_spectral_flatness"
        ),
        "acoustic_burst_high_frequency_fraction": acoustic_candidate.get(
            "burst_high_frequency_fraction"
        ),
        "acoustic_spectral_flux_z": acoustic_candidate.get("spectral_flux_z"),
        "acoustic_high_frequency_rise_db": acoustic_candidate.get(
            "high_frequency_rise_db"
        ),
        "acoustic_energy_rise_db": acoustic_candidate.get("energy_rise_db"),
        "acoustic_energy_slope_12ms_db": acoustic_candidate.get(
            "energy_slope_12ms_db"
        ),
        "acoustic_distance_from_anchor_ms": acoustic_candidate.get(
            "distance_from_anchor_ms"
        ),
        "acoustic_candidate_attribution": acoustic_candidate.get("attribution"),
        "acoustic_target_window_s": acoustic.get("target_window_s"),
        "acoustic_top_candidate_time_s": acoustic.get("top_candidate_time_s"),
        "acoustic_target_occurrence_index": acoustic.get("target_occurrence_index"),
        "acoustic_target_occurrence_count": acoustic.get("target_occurrence_count"),
        "visual_contact_time_s": visual.get("contact_time_s"),
        "visual_candidate_time_s": visual.get("candidate_time_s"),
        "visual_release_time_s": visual.get("release_time_s"),
        "visual_closure_duration_ms": visual.get("closure_duration_ms"),
        "visual_timing_uncertainty_ms": visual.get("timing_uncertainty_ms"),
        "visual_valid_frame_fraction": visual.get("valid_frame_fraction"),
        "visual_quality_score": visual.get("quality_score"),
        "visual_measurable": visual.get("measurable"),
        "visual_exclusion_reasons": visual.get("exclusion_reasons"),
        "mouth_shape_category_at_burst": shape.get("category"),
        "mouth_shape_frame_time_s": shape.get("frame_time_s"),
        "mouth_shape_frame_error_ms": shape.get("frame_time_error_ms"),
        "mouth_shape_pair_apertures": shape.get("pair_apertures"),
        "mouth_shape_median_aperture": shape.get("median_aperture"),
        "mouth_shape_maximum_aperture": shape.get("maximum_aperture"),
        "mouth_shape_aperture_spread": shape.get("aperture_spread"),
        "mouth_shape_width_px": shape.get("mouth_width_px"),
        "mouth_shape_face_quality": shape.get("face_quality"),
        "mouth_shape_repeated_frame": shape.get("repeated_frame"),
        "mouth_shape_valid": shape.get("valid"),
        "mouth_shape_exclusion_reasons": shape.get("exclusion_reasons"),
        "top_candidate_mouth_shape_category": diagnostic_shape.get("category"),
        "top_candidate_mouth_shape_frame_time_s": diagnostic_shape.get("frame_time_s"),
        "top_candidate_mouth_shape_frame_error_ms": diagnostic_shape.get(
            "frame_time_error_ms"
        ),
        "top_candidate_mouth_shape_pair_apertures": diagnostic_shape.get(
            "pair_apertures"
        ),
        "top_candidate_mouth_shape_median_aperture": diagnostic_shape.get(
            "median_aperture"
        ),
        "top_candidate_mouth_shape_maximum_aperture": diagnostic_shape.get(
            "maximum_aperture"
        ),
        "top_candidate_mouth_shape_valid": diagnostic_shape.get("valid"),
        "top_candidate_mouth_shape_exclusion_reasons": diagnostic_shape.get(
            "exclusion_reasons"
        ),
        "raw_lip_sample_count": lip_sample_count,
        "processing_error": record.get("processing_error"),
    }


def comparison_csv_row(
    spec: AnalysisSpec, comparison: Mapping[str, object]
) -> dict[str, object]:
    ci = comparison.get("bootstrap_ci95_ms")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        ci_low, ci_high = ci
    else:
        ci_low = ci_high = None
    return {
        "analysis": spec.key,
        "inferential_role": spec.inferential_role,
        "phone_classes": spec.phone_classes,
        "candidate_epoch_count": comparison.get("candidate_epoch_count"),
        "control_epoch_count": comparison.get("control_epoch_count"),
        "candidate_event_count": comparison.get("candidate_event_count"),
        "control_event_count": comparison.get("control_event_count"),
        "candidate_median_ms": comparison.get("candidate_median_ms"),
        "control_median_ms": comparison.get("control_median_ms"),
        "median_difference_ms": comparison.get("median_difference_ms"),
        "median_difference_frames": comparison.get("median_difference_frames"),
        "bootstrap_ci95_low_ms": ci_low,
        "bootstrap_ci95_high_ms": ci_high,
        "probability_candidate_greater": comparison.get(
            "probability_candidate_greater"
        ),
        "cliffs_delta": comparison.get("cliffs_delta"),
        "exploratory_one_sided_permutation_p": comparison.get(
            "exploratory_one_sided_permutation_p"
        ),
        "candidate_fraction_epochs_over_margin": comparison.get(
            "candidate_fraction_epochs_over_margin"
        ),
        "control_fraction_epochs_over_margin": comparison.get(
            "control_fraction_epochs_over_margin"
        ),
        "margin_ms": comparison.get("margin_ms"),
        "cautions": comparison.get("cautions"),
    }


def _selection_record(token: BilabialToken) -> dict[str, object]:
    reasons = [token.exclusion_reason] if token.exclusion_reason else []
    return {
        "selection": token.as_dict(),
        "attempted": False,
        "status": "selection_excluded" if not token.eligible else "pending",
        "measurable": False,
        "release_lag_ms": None,
        "diagnostic_release_lag_ms": None,
        "diagnostic_candidate_lag_ms": None,
        "acoustic_auto_accepted": False,
        "acoustic_auto_acceptance_reasons": reasons,
        "exclusion_reasons": reasons,
        "measurement": None,
        "diagnostic_mouth_shape_at_top_acoustic_candidate": None,
        "audio_window": None,
        "lip_samples": [],
        "processing_error": None,
    }


def _deferred_record(token: BilabialToken) -> dict[str, object]:
    record = _selection_record(token)
    record.update(
        status="deferred_by_max_events",
        exclusion_reasons=["not_processed_max_events_limit"],
    )
    return record


def _measure_token(
    token: BilabialToken,
    *,
    video: Path,
    face_model: Path,
    roi: tuple[float, float, float, float] | None,
    interview_end_s: float,
    acoustic_config: AcousticReleaseConfig,
    visual_config: VisualReleaseConfig,
) -> dict[str, object]:
    start_s = max(0.0, token.anchor_s - EVENT_WINDOW_BEFORE_S)
    end_s = min(interview_end_s, token.anchor_s + EVENT_WINDOW_AFTER_S)
    audio = decode_audio_window(
        video,
        start_s=start_s,
        end_s=end_s,
        sample_rate=AUDIO_SAMPLE_RATE,
    )
    # The ASR word spans bound *attribution* only: a burst inside a
    # neighbouring word's span is not reported as this token's release.
    acoustic = estimate_acoustic_release(
        audio.waveform,
        audio.sample_rate,
        window_start_s=audio.start_s,
        anchor_time_s=token.anchor_s,
        phone_class=token.phoneme_class,
        coverage_mask=audio.coverage_mask,
        config=acoustic_config,
        target_window_s=(token.token_start_s, token.token_end_s),
        previous_window_s=token.previous_word_window_s,
        next_window_s=token.next_word_window_s,
        target_occurrence_index=token.word_occurrence_index,
        target_occurrence_anchors_s=token.word_occurrence_anchors_s or None,
    )

    # The visual search receives the ASR anchor, not the acoustic estimate.
    # This keeps the two release measurements independent until combination.
    lip_samples = extract_native_lip_samples(
        video,
        face_model,
        start_s=start_s,
        end_s=end_s,
        roi=roi,
    )
    visual = estimate_visual_release(
        lip_samples, anchor_time_s=token.anchor_s, config=visual_config
    )
    measurement = combine_release_estimates(
        acoustic, visual, lip_samples, visual_config=visual_config
    )
    diagnostic_candidate_lag_ms = (
        (visual.candidate_time_s - acoustic.candidate_time_s) * 1000.0
        if visual.candidate_time_s is not None and acoustic.candidate_time_s is not None
        else None
    )
    if acoustic.candidate_time_s is not None:
        diagnostic_acoustic = replace(
            acoustic,
            release_time_s=acoustic.candidate_time_s,
            measurable=True,
        )
        diagnostic_shape = identify_mouth_shape_at_acoustic_release(
            lip_samples, diagnostic_acoustic, config=visual_config
        )
    else:
        diagnostic_shape = None
    auto_acceptance_reasons = list(acoustic.exclusion_reasons)
    if token.asr_probability < CONSERVATIVE_MINIMUM_ASR_PROBABILITY:
        auto_acceptance_reasons.append("below_conservative_asr_probability")
    acoustic_auto_accepted = acoustic.measurable and not auto_acceptance_reasons
    final_measurable = measurement.measurable and acoustic_auto_accepted
    final_reasons = list(
        dict.fromkeys((*measurement.exclusion_reasons, *auto_acceptance_reasons))
    )
    return {
        "selection": token.as_dict(),
        "attempted": True,
        "status": "measurable" if final_measurable else "measurement_excluded",
        "measurable": final_measurable,
        "release_lag_ms": measurement.lag_ms if final_measurable else None,
        "diagnostic_release_lag_ms": measurement.lag_ms,
        "diagnostic_candidate_lag_ms": diagnostic_candidate_lag_ms,
        "acoustic_auto_accepted": acoustic_auto_accepted,
        "acoustic_auto_acceptance_reasons": auto_acceptance_reasons,
        "exclusion_reasons": final_reasons,
        "measurement": measurement.as_dict(),
        "diagnostic_mouth_shape_at_top_acoustic_candidate": (
            diagnostic_shape.as_dict() if diagnostic_shape is not None else None
        ),
        "audio_window": {
            "start_s": audio.start_s,
            "end_s": audio.end_s,
            "sample_rate": audio.sample_rate,
            "sample_count": len(audio.waveform),
            "coverage_fraction": audio.coverage_fraction,
            "warnings": list(audio.warnings),
        },
        "lip_samples": lip_samples_as_records(lip_samples),
        "processing_error": None,
    }


def _processing_error_record(token: BilabialToken, error: Exception) -> dict[str, object]:
    record = _selection_record(token)
    reason = f"processing_error:{type(error).__name__}"
    record.update(
        attempted=True,
        status="processing_error",
        exclusion_reasons=[reason],
        processing_error={"type": type(error).__name__, "message": str(error)},
    )
    return record


def _measurement_configuration(
    *,
    video: Path,
    words_json: Path,
    intervals_json: Path,
    speaker_token_manifest: Path,
    face_model: Path,
    roi: tuple[float, float, float, float] | None,
    interview_end_s: float,
    interview_end_source: str,
    acoustic_config: AcousticReleaseConfig,
    visual_config: VisualReleaseConfig,
    token_inventory_scope: str,
    frame_timing: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "video": _path_fingerprint(video),
            "words_json": _path_fingerprint(words_json),
            "intervals_json": _path_fingerprint(intervals_json),
            "speaker_token_manifest": _path_fingerprint(speaker_token_manifest),
            "face_model": _path_fingerprint(face_model),
        },
        "implementation": {
            "runner": _path_fingerprint(Path(__file__)),
            "artifact_io": _path_fingerprint(
                PROJECT / "video_integrity_analyzer" / "artifact_io.py"
            ),
            "plosive_sync": _path_fingerprint(
                PROJECT / "video_integrity_analyzer" / "plosive_sync.py"
            ),
            "plosive_manifest": _path_fingerprint(
                PROJECT / "video_integrity_analyzer" / "plosive_manifest.py"
            ),
            "plosive_stats": _path_fingerprint(
                PROJECT / "video_integrity_analyzer" / "plosive_stats.py"
            ),
        },
        "protocol": {
            "interview_end_s": interview_end_s,
            "interview_end_source": interview_end_source,
            "minimum_asr_probability": MINIMUM_ASR_PROBABILITY,
            "conservative_minimum_asr_probability": (
                CONSERVATIVE_MINIMUM_ASR_PROBABILITY
            ),
            "boundary_guard_s": BOUNDARY_GUARD_S,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "event_window_before_s": EVENT_WINDOW_BEFORE_S,
            "event_window_after_s": EVENT_WINDOW_AFTER_S,
            "roi": list(roi) if roi is not None else None,
            "acoustic_release_config": asdict(acoustic_config),
            "visual_release_config": asdict(visual_config),
            "token_inventory_scope": token_inventory_scope,
            "supported_media_profile": SUPPORTED_MEDIA_PROFILE,
            "source_frame_timing": dict(frame_timing),
            "selection_statement": (
                "ASR selects all explicit Japanese /p/ and /b/ kana before visual inspection; "
                "waveform and native-PTS lip geometry independently refine release times."
            ),
        },
    }


def _load_checkpoint(
    checkpoint_path: Path,
    expected_configuration: Mapping[str, object],
    *,
    resume: bool,
) -> dict[str, dict[str, object]]:
    if not resume or not checkpoint_path.exists():
        return {}
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if data.get("configuration") != expected_configuration:
        raise RuntimeError(
            f"{checkpoint_path} belongs to different inputs or measurement settings; "
            "use --no-resume or a different output directory"
        )
    raw = data.get("records_by_event_id", {})
    if not isinstance(raw, dict):
        raise RuntimeError(f"malformed checkpoint: {checkpoint_path}")
    return {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, dict) and value.get("attempted") is True
    }


def _checkpoint_payload(
    *,
    configuration: Mapping[str, object],
    tokens: Sequence[BilabialToken],
    processed: Mapping[str, Mapping[str, object]],
    target_ids: set[str],
) -> dict[str, object]:
    eligible_ids = {token.event_id for token in tokens if token.eligible}
    completed_target = len(target_ids.intersection(processed))
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": _now_utc(),
        "configuration": configuration,
        "progress": {
            "tokens_total": len(tokens),
            "eligible_total": len(eligible_ids),
            "target_event_count": len(target_ids),
            "completed_target_count": completed_target,
            "complete_for_requested_target": completed_target == len(target_ids),
        },
        "records_by_event_id": dict(processed),
    }


def _materialize_records(
    tokens: Sequence[BilabialToken],
    processed: Mapping[str, Mapping[str, object]],
    target_ids: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for token in tokens:
        if token.event_id in processed:
            records.append(dict(processed[token.event_id]))
        elif not token.eligible:
            records.append(_selection_record(token))
        elif token.event_id not in target_ids:
            records.append(_deferred_record(token))
        else:
            records.append(_selection_record(token))
    return records


def audit_counts(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Count selection and measurement disposition by group and speaker."""

    overall = Counter()
    by_group: dict[str, Counter[str]] = {}
    by_speaker: dict[str, Counter[str]] = {}
    exclusions = Counter()
    accepted_mouth_shapes = Counter()
    diagnostic_mouth_shapes = Counter()
    for record in records:
        selection = _nested(record, "selection")
        group = str(selection.get("group") or "unassigned")
        speaker = str(selection.get("speaker") or "unassigned")
        counters = (overall, by_group.setdefault(group, Counter()), by_speaker.setdefault(speaker, Counter()))
        visual = _nested(_nested(record, "measurement"), "visual")
        for counter in counters:
            counter["selected"] += 1
            counter["eligible"] += int(selection.get("eligible") is True)
            counter["attempted"] += int(record.get("attempted") is True)
            counter["acoustic_auto_accepted"] += int(
                record.get("acoustic_auto_accepted") is True
            )
            counter["visual_release_measurable"] += int(
                visual.get("measurable") is True
            )
            counter["measurable"] += int(record.get("measurable") is True)
        reasons = record.get("exclusion_reasons")
        if isinstance(reasons, (list, tuple)):
            exclusions.update(str(reason) for reason in reasons if reason)
        shape = _nested(_nested(record, "measurement"), "mouth_shape_at_acoustic_release")
        if record.get("acoustic_auto_accepted") is True and shape.get("category"):
            accepted_mouth_shapes[str(shape["category"])] += 1
        diagnostic_shape = _nested(
            record, "diagnostic_mouth_shape_at_top_acoustic_candidate"
        )
        if (
            record.get("acoustic_auto_accepted") is not True
            and diagnostic_shape.get("category")
        ):
            diagnostic_mouth_shapes[str(diagnostic_shape["category"])] += 1
    return {
        "overall": dict(overall),
        "by_group": {key: dict(value) for key, value in sorted(by_group.items())},
        "by_speaker": {key: dict(value) for key, value in sorted(by_speaker.items())},
        "exclusion_reasons": dict(exclusions.most_common()),
        "mouth_shape_at_accepted_acoustic_release": dict(
            accepted_mouth_shapes.most_common()
        ),
        "mouth_shape_at_rejected_top_candidate_diagnostic": dict(
            diagnostic_mouth_shapes.most_common()
        ),
    }


def _events_for_spec(
    records: Sequence[Mapping[str, object]], spec: AnalysisSpec
) -> list[dict[str, object]]:
    return [
        dict(record)
        for record in records
        if _nested(record, "selection").get("phoneme_class") in spec.phone_classes
    ]


def _event_for_stats(record: Mapping[str, object]) -> dict[str, object]:
    selection = _nested(record, "selection")
    return {
        "speaker": selection.get("speaker"),
        "group": selection.get("group"),
        "epoch_id": selection.get("epoch_id"),
        "release_lag_ms": record.get("release_lag_ms"),
        "measurable": record.get("measurable") is True,
    }


def _plot_analysis(
    records: Sequence[Mapping[str, object]],
    epochs: Sequence[Mapping[str, object]],
    *,
    margin_ms: float,
    label: str,
    output_path: Path,
) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # optional dependency
        return f"plot unavailable: {type(error).__name__}: {error}"

    measurable = [record for record in records if record.get("measurable") is True]
    groups = ("candidate", "control")
    colors = {"candidate": "#D55E00", "control": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(label)

    axis = axes[0, 0]
    for group in groups:
        selected = [
            record
            for record in measurable
            if _nested(record, "selection").get("group") == group
        ]
        axis.scatter(
            [float(_nested(record, "selection")["anchor_s"]) / 60.0 for record in selected],
            [float(record["release_lag_ms"]) for record in selected],
            s=28,
            alpha=0.8,
            label=group,
            color=colors[group],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(margin_ms, color="#666666", linestyle="--", linewidth=0.8)
    axis.set(xlabel="Interview time (min)", ylabel="Visible release lag (ms)")
    axis.set_title("Measurable events")
    axis.legend(loc="best")

    axis = axes[0, 1]
    epoch_values = {
        group: [
            float(epoch["median_lag_ms"])
            for epoch in epochs
            if epoch.get("group") == group
        ]
        for group in groups
    }
    nonempty_groups = [group for group in groups if epoch_values[group]]
    if nonempty_groups:
        axis.boxplot(
            [epoch_values[group] for group in nonempty_groups],
            tick_labels=nonempty_groups,
            showmeans=True,
        )
        for position, group in enumerate(nonempty_groups, start=1):
            count = len(epoch_values[group])
            offsets = [0.0] if count == 1 else [(-0.10 + 0.20 * i / (count - 1)) for i in range(count)]
            axis.scatter(
                [position + offset for offset in offsets],
                epoch_values[group],
                color=colors[group],
                s=24,
                alpha=0.75,
            )
    else:
        axis.text(0.5, 0.5, "No measurable epochs", ha="center", va="center")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(margin_ms, color="#666666", linestyle="--", linewidth=0.8)
    axis.set(ylabel="Epoch median lag (ms)")
    axis.set_title("Synchronization epochs (comparison unit)")

    axis = axes[1, 0]
    count_matrix = []
    for group in groups:
        selected = [
            record
            for record in records
            if _nested(record, "selection").get("group") == group
        ]
        count_matrix.append(
            (
                sum(_nested(record, "selection").get("eligible") is True for record in selected),
                sum(record.get("acoustic_auto_accepted") is True for record in selected),
                sum(
                    _nested(_nested(record, "measurement"), "visual").get("measurable")
                    is True
                    for record in selected
                ),
                sum(record.get("measurable") is True for record in selected),
            )
        )
    x = [0, 1]
    width = 0.18
    for offset, index, name in (
        (-1.5 * width, 0, "eligible"),
        (-0.5 * width, 1, "acoustic accepted"),
        (0.5 * width, 2, "visual measurable"),
        (1.5 * width, 3, "combined measurable"),
    ):
        axis.bar([value + offset for value in x], [row[index] for row in count_matrix], width=width, label=name)
    axis.set_xticks(x, groups)
    axis.set(ylabel="Event count")
    axis.set_title("Audit disposition")
    axis.legend(loc="best")

    axis = axes[1, 1]
    categories = ("closed/contact", "transition", "open", "unavailable")
    bottoms = [0, 0]
    for category in categories:
        values = []
        for group in groups:
            values.append(
                sum(
                    record.get("acoustic_auto_accepted") is True
                    and _nested(
                        _nested(record, "measurement"),
                        "mouth_shape_at_acoustic_release",
                    ).get("category")
                    == category
                    for record in records
                    if _nested(record, "selection").get("group") == group
                )
            )
        axis.bar(x, values, bottom=bottoms, label=category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_xticks(x, groups)
    axis.set(ylabel="Auto-accepted acoustic event count")
    axis.set_title("Mouth shape at accepted acoustic release")
    axis.legend(loc="best", fontsize="small")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return None


def _build_analysis(
    spec: AnalysisSpec,
    records: Sequence[Mapping[str, object]],
    *,
    fps: float,
    margin_ms: float,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    selected = _events_for_spec(records, spec)
    epoch_estimates = summarize_epochs(
        (_event_for_stats(record) for record in selected), margin_ms=margin_ms
    )
    comparison = compare_candidate_to_controls(
        epoch_estimates,
        fps=fps,
        margin_ms=margin_ms,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        seed=seed,
    )
    epoch_records = [epoch.as_dict() for epoch in epoch_estimates]
    comparison_record = comparison.as_dict()
    analysis = {
        "analysis": asdict(spec),
        "counts": audit_counts(selected),
        "event_ids": [
            str(_nested(record, "selection").get("event_id")) for record in selected
        ],
        "epoch_estimates": epoch_records,
        "comparison": comparison_record,
    }
    return analysis, epoch_records, comparison_record


def _parse_roi(values: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    roi = tuple(float(value) for value in values)
    if len(roi) != 4 or not all(math.isfinite(value) for value in roi):
        raise ValueError("ROI must contain four finite normalized values")
    x0, y0, x1, y1 = roi
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return roi  # type: ignore[return-value]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _resolve_interview_end(
    requested_end_s: float | None, *, media_duration_s: float
) -> float:
    """Resolve the analysis cutoff without embedding a recording-specific time."""

    if not math.isfinite(media_duration_s) or media_duration_s <= 0.0:
        raise ValueError("media duration is unavailable")
    end_s = media_duration_s if requested_end_s is None else float(requested_end_s)
    if not math.isfinite(end_s) or end_s <= 0.0:
        raise ValueError("interview end must be a positive finite time")
    if end_s > media_duration_s + 1e-6:
        raise ValueError(
            f"interview end {end_s:.6f}s exceeds media duration "
            f"{media_duration_s:.6f}s"
        )
    return min(end_s, media_duration_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--words-json", required=True, type=Path)
    parser.add_argument("--intervals-json", required=True, type=Path)
    parser.add_argument(
        "--speaker-token-manifest",
        required=True,
        type=Path,
        help="finalized token-level on-screen name-label guard manifest",
    )
    parser.add_argument("--face-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--interview-end",
        type=float,
        default=None,
        help=(
            "interview analysis cutoff in seconds; use this to exclude post-call "
            "material (default: probed media duration)"
        ),
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="optional normalized active-speaker face ROI",
    )
    parser.add_argument(
        "--max-events",
        type=_positive_integer,
        help="process only the first N eligible events (deterministic smoke test)",
    )
    parser.add_argument(
        "--bootstrap-iterations", type=_positive_integer, default=20_000
    )
    parser.add_argument(
        "--permutation-iterations", type=_positive_integer, default=100_000
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--allow-incomplete-token-inventory",
        action="store_true",
        help=(
            "diagnostic-only override for literal-kana inventories; outputs must not "
            "be described as complete and must not drive an all-cases evidence video"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore and replace an existing event checkpoint",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    video = args.video.expanduser().resolve()
    words_json = args.words_json.expanduser().resolve()
    intervals_json = args.intervals_json.expanduser().resolve()
    speaker_token_manifest = args.speaker_token_manifest.expanduser().resolve()
    face_model = args.face_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (
        video,
        words_json,
        intervals_json,
        speaker_token_manifest,
        face_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi = _parse_roi(args.roi)
    face_preflight = preflight_face_runtime(
        model_path=face_model,
        output_dir=output_dir / "face_runtime_preflight",
    )
    print(
        "Face runtime preflight passed "
        f"({face_preflight['processed_frames']} synthetic frame).",
        flush=True,
    )
    media = probe_media(video)
    _validate_supported_media_profile(media)
    frame_timing_diagnostics = probe_video_frame_timing(video, expected_fps=24.0)
    if not frame_timing_diagnostics.is_cfr:
        raise ValueError(
            "fixed acoustic/visual gates require strict 24fps CFR source PTS; "
            f"timing diagnostics: {frame_timing_diagnostics.as_dict()}"
        )
    acoustic_config = AcousticReleaseConfig.conservative()
    visual_config = VisualReleaseConfig()
    fps = float(media.average_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("media average frame rate is unavailable")
    margin_ms = 2_000.0 / fps
    interview_end_s = _resolve_interview_end(
        args.interview_end, media_duration_s=float(media.duration_s)
    )

    intervals = load_speaker_intervals(intervals_json)
    tokens = extract_bilabial_tokens(
        words_json,
        intervals,
        interview_end_s=interview_end_s,
        minimum_asr_probability=MINIMUM_ASR_PROBABILITY,
        boundary_guard_s=BOUNDARY_GUARD_S,
    )
    manifest_data = json.loads(speaker_token_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, Mapping):
        raise ValueError("speaker-token manifest root must be an object")
    method = manifest_data.get("method")
    token_inventory_scope = (
        str(method.get("token_inventory_scope", "missing"))
        if isinstance(method, Mapping)
        else "missing"
    )
    if token_inventory_scope != "reading_complete" and not args.allow_incomplete_token_inventory:
        raise ValueError(
            "complete analysis requires a speaker-token manifest with "
            "method.token_inventory_scope='reading_complete'; rebuild it with "
            "word-level Japanese readings or use --allow-incomplete-token-inventory "
            "only for explicitly incomplete diagnostics"
        )
    tokens, speaker_join_by_event, speaker_join_audit = enforce_speaker_token_manifest(
        tokens, manifest_data
    )
    eligible = [token for token in tokens if token.eligible]
    target = eligible[: args.max_events] if args.max_events is not None else eligible
    target_ids = {token.event_id for token in target}

    configuration = _measurement_configuration(
        video=video,
        words_json=words_json,
        intervals_json=intervals_json,
        speaker_token_manifest=speaker_token_manifest,
        face_model=face_model,
        roi=roi,
        interview_end_s=interview_end_s,
        interview_end_source=(
            "media_duration" if args.interview_end is None else "command_line"
        ),
        acoustic_config=acoustic_config,
        visual_config=visual_config,
        token_inventory_scope=token_inventory_scope,
        frame_timing=frame_timing_diagnostics.as_dict(),
    )
    checkpoint_path = output_dir / "checkpoint.json"
    processed = _load_checkpoint(
        checkpoint_path, configuration, resume=not args.no_resume
    )
    # A larger earlier run may contain valid records outside this smoke-test
    # target. They stay in the checkpoint but are omitted from current outputs.
    completed_before = len(target_ids.intersection(processed))
    print(
        f"Preselected {len(tokens)} tokens; {len(eligible)} eligible; "
        f"target {len(target)}; resuming {completed_before} completed.",
        flush=True,
    )

    for index, token in enumerate(target, start=1):
        if token.event_id in processed:
            continue
        print(
            f"[{index}/{len(target)}] {token.event_id} {token.speaker} "
            f"/{token.phoneme_class}/ at {token.anchor_s:.3f}s",
            flush=True,
        )
        try:
            record = _measure_token(
                token,
                video=video,
                face_model=face_model,
                roi=roi,
                interview_end_s=interview_end_s,
                acoustic_config=acoustic_config,
                visual_config=visual_config,
            )
        except Exception as error:  # one bad event must not destroy the audit run
            record = _processing_error_record(token, error)
            print(
                f"  excluded after {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        processed[token.event_id] = record
        _write_json(
            checkpoint_path,
            _checkpoint_payload(
                configuration=configuration,
                tokens=tokens,
                processed=processed,
                target_ids=target_ids,
            ),
        )

    if not checkpoint_path.exists() or not target:
        _write_json(
            checkpoint_path,
            _checkpoint_payload(
                configuration=configuration,
                tokens=tokens,
                processed=processed,
                target_ids=target_ids,
            ),
        )

    current_processed = {
        event_id: record
        for event_id, record in processed.items()
        if event_id in target_ids
    }
    records = _materialize_records(tokens, current_processed, target_ids)
    for record in records:
        event_id = str(_nested(record, "selection").get("event_id"))
        record["speaker_token_manifest_join"] = speaker_join_by_event.get(event_id)
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _now_utc(),
        "configuration": configuration,
        "media": {
            "duration_s": media.duration_s,
            "width": media.width,
            "height": media.height,
            "average_fps": fps,
            "audio_sample_rate": media.audio_sample_rate,
        },
        "requested_max_events": args.max_events,
        "speaker_token_manifest_join_counts": speaker_join_audit["counts"],
        "statistics": {
            "comparison_unit": "synchronization epoch median",
            "two_frame_practical_margin_ms": margin_ms,
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "seed": args.seed,
            "positive_lag_definition": "visible release occurs after acoustic release",
        },
        "limitations": [
            "A non-zero lag is not a deepfake or identity classifier.",
            "Speaker is confounded with device, network path, conferencing jitter buffer, and encoder path.",
            "The exploratory label-permutation p-value does not make participant streams randomized or exchangeable.",
            "ASR selection errors and /b/ lenition remain possible; /p/ is the predeclared primary stratum.",
        ],
    }

    _write_json(output_dir / "events.json", {**run_metadata, "events": records})
    _write_json(
        output_dir / "speaker_token_manifest_join.json",
        {**run_metadata, "join_audit": speaker_join_audit},
    )
    _write_csv(output_dir / "events.csv", [event_csv_row(record) for record in records])

    analyses: dict[str, object] = {}
    comparison_rows: list[dict[str, object]] = []
    all_epoch_rows: list[dict[str, object]] = []
    plot_warnings: list[str] = []
    for offset, spec in enumerate(analysis_definitions()):
        analysis, epochs, comparison = _build_analysis(
            spec,
            records,
            fps=fps,
            margin_ms=margin_ms,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            seed=args.seed + offset * 1_000,
        )
        analyses[spec.key] = analysis
        stratum_dir = output_dir / spec.directory
        stratum_records = _events_for_spec(records, spec)
        _write_json(stratum_dir / "analysis.json", {**run_metadata, **analysis})
        _write_csv(
            stratum_dir / "events.csv",
            [event_csv_row(record) for record in stratum_records],
        )
        _write_csv(stratum_dir / "epochs.csv", epochs)
        comparison_row = comparison_csv_row(spec, comparison)
        _write_csv(stratum_dir / "comparison.csv", [comparison_row])
        comparison_rows.append(comparison_row)
        all_epoch_rows.extend({"analysis": spec.key, **epoch} for epoch in epochs)
        if not args.skip_plots:
            warning = _plot_analysis(
                stratum_records,
                epochs,
                margin_ms=margin_ms,
                label=spec.label,
                output_path=stratum_dir / "diagnostic.png",
            )
            if warning:
                plot_warnings.append(f"{spec.key}: {warning}")

    target_records = [
        record
        for record in records
        if str(_nested(record, "selection").get("event_id")) in target_ids
    ]
    target_processing_errors = sum(
        record.get("status") == "processing_error" for record in target_records
    )
    run_health = {
        "status": (
            "FAIL_ALL_TARGET_PROCESSING_ERRORS"
            if target_records and target_processing_errors == len(target_records)
            else "PASS"
        ),
        "target_count": len(target_records),
        "target_processing_error_count": target_processing_errors,
    }
    summary = {
        **run_metadata,
        "run_health": run_health,
        "counts": audit_counts(records),
        "analyses": analyses,
        "plot_warnings": plot_warnings,
        "artifacts": {
            "event_audit_json": "events.json",
            "event_audit_csv": "events.csv",
            "checkpoint": "checkpoint.json",
            "speaker_token_manifest_join": "speaker_token_manifest_join.json",
            "comparisons_csv": "comparisons.csv",
            "epochs_csv": "epoch_estimates.csv",
            "strata": {
                spec.key: {
                    "analysis_json": f"{spec.directory}/analysis.json",
                    "events_csv": f"{spec.directory}/events.csv",
                    "epochs_csv": f"{spec.directory}/epochs.csv",
                    "comparison_csv": f"{spec.directory}/comparison.csv",
                    "diagnostic_plot": (
                        None if args.skip_plots else f"{spec.directory}/diagnostic.png"
                    ),
                }
                for spec in analysis_definitions()
            },
        },
    }
    _write_json(output_dir / "analysis.json", summary)
    _write_csv(output_dir / "comparisons.csv", comparison_rows)
    _write_csv(output_dir / "epoch_estimates.csv", all_epoch_rows)
    print(f"Wrote analysis bundle to {output_dir}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if _nested(summary, "run_health").get("status") != "PASS":
        print(
            "error: every requested target ended in processing_error; "
            "inspect analysis.json and face_worker.log",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
