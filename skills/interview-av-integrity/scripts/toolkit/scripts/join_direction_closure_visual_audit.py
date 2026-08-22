#!/usr/bin/env python3
"""Unblind and join visual-contact annotations to machine output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_direction_closure_blind_sheets import normalized_review_value


def read_keyed_csv(path: Path, *, label: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {str(row.get("blind_id", "")).strip(): row for row in rows}
    if "" in keyed or not keyed:
        raise RuntimeError(f"{label} must contain nonempty blind_id values")
    if len(keyed) != len(rows):
        raise RuntimeError(f"Duplicate blind_id in {label}")
    return keyed


def first_value(row: dict[str, str], *columns: str, default: str = "") -> str:
    populated = [str(row.get(column, "")).strip() for column in columns if str(row.get(column, "")).strip()]
    if len(set(populated)) > 1:
        raise RuntimeError(f"Conflicting values in columns: {', '.join(columns)}")
    return populated[0] if populated else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--followup", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    event_rows = payload["events"]
    event_by_id = {event["runner_event_id"]: event for event in event_rows}
    if not event_by_id or len(event_by_id) != len(event_rows):
        raise RuntimeError("Events must contain unique, nonempty runner_event_id values")
    blind_key = json.loads(args.blind_key.read_text(encoding="utf-8"))["events"]
    key_by_id = {row["blind_id"]: row for row in blind_key}
    if not key_by_id or len(key_by_id) != len(blind_key):
        raise RuntimeError("Blind key must contain unique, nonempty blind IDs")
    keyed_runner_ids = [str(row.get("runner_event_id", "")).strip() for row in blind_key]
    if any(not event_id for event_id in keyed_runner_ids) or len(
        set(keyed_runner_ids)
    ) != len(keyed_runner_ids):
        raise RuntimeError("Blind key must map one-to-one to unique runner event IDs")
    annotation_by_id = read_keyed_csv(args.annotations, label="annotations")
    if set(annotation_by_id) != set(key_by_id):
        raise RuntimeError("Annotations must contain exactly the blind-key IDs")

    followup_by_id: dict[str, dict[str, str]] = {}
    if args.followup:
        followup_by_id = read_keyed_csv(args.followup, label="follow-up annotations")
        unknown_followup = sorted(set(followup_by_id) - set(key_by_id))
        if unknown_followup:
            raise RuntimeError(
                "Follow-up annotations contain unknown blind IDs: "
                + ", ".join(unknown_followup)
            )

    rows: list[dict[str, object]] = []
    for order, key_row in enumerate(blind_key, start=1):
        blind_id = key_row["blind_id"]
        annotation = annotation_by_id[blind_id]
        followup = followup_by_id.get(blind_id)
        event = event_by_id[key_row["runner_event_id"]]
        geometry = event["geometry"]
        machine_category = geometry["closure_category"]
        human_contact = normalized_review_value(annotation, "contact")
        human_timing = normalized_review_value(annotation, "timing_relative_audio")
        human_category = normalized_review_value(
            annotation, "machine_window_category"
        )
        if human_category == "ambiguous":
            category_agreement = "ambiguous"
            presence_agreement = "ambiguous"
        else:
            category_agreement = "match" if human_category == machine_category else "mismatch"
            human_positive = human_category in {"weak_contact", "strong_contact"}
            machine_positive = machine_category in {"weak_contact", "strong_contact"}
            presence_agreement = "match" if human_positive == machine_positive else "mismatch"
        human_post = first_value(
            annotation,
            "post_audio_only_contact",
            "human_post_audio_only_contact",
            "human_post_only_gt_41_7ms",
            default="not_applicable",
        ).lower()
        if human_post in {"true", "false"} and human_category in {"weak_contact", "strong_contact"}:
            post_only_agreement = (
                "match"
                if (human_post == "true") is (geometry["post_audio_only_contact"] is True)
                else "mismatch"
            )
        elif human_post == "ambiguous":
            post_only_agreement = "ambiguous"
        else:
            post_only_agreement = "not_applicable"
        selection = event["selection"]
        rows.append(
            {
                "blind_order": order,
                "blind_id": blind_id,
                "runner_event_id": event["runner_event_id"],
                "speaker": selection["speaker"],
                "group": selection["group"],
                "epoch_id": selection["epoch_id"],
                "phoneme_class": selection["phoneme_class"],
                "kana": selection.get("kana", ""),
                "token_text": selection.get("token_text", ""),
                "audio_release_time_s": event["audio_release_time_s"],
                "human_contact": human_contact,
                "human_timing_relative_audio": human_timing,
                "human_first_clear_contact_offset_ms": first_value(
                    annotation, "first_clear_contact_offset_ms"
                ),
                "human_machine_window_category": human_category,
                "human_post_audio_only_contact": human_post,
                "human_confidence": first_value(annotation, "confidence"),
                "human_blind_note": first_value(annotation, "blind_note", "note"),
                "human_followup_first_contact_offset_ms": (
                    first_value(
                        followup,
                        "first_contact_followup_offset_ms",
                        "human_first_contact_through_plus500_ms",
                    )
                    if followup
                    else first_value(annotation, "first_clear_contact_offset_ms")
                ),
                "human_release_after_contact": (
                    first_value(
                        followup,
                        "release_after_contact",
                        "human_release_after_contact",
                    )
                    if followup
                    else "not_extended_review_target"
                ),
                "human_release_after_contact_offset_ms": (
                    first_value(
                        followup,
                        "release_after_contact_offset_ms",
                        "human_release_after_contact_offset_ms",
                    )
                    if followup
                    else ""
                ),
                "human_followup_confidence": (
                    first_value(followup, "followup_confidence", "confidence")
                    if followup
                    else "not_reviewed"
                ),
                "human_followup_note": (
                    first_value(followup, "followup_note", "note")
                    if followup
                    else ""
                ),
                "machine_closure_category": machine_category,
                "machine_contact_frame_count": geometry["contact_nonrepeated_frame_count"],
                "machine_contact_frame_offsets_ms": json.dumps(
                    geometry["contact_frame_offsets_ms"], separators=(",", ":")
                ),
                "machine_post_audio_only_contact": str(
                    geometry["post_audio_only_contact"]
                ).lower(),
                "category_agreement": category_agreement,
                "contact_presence_agreement": presence_agreement,
                "post_only_agreement": post_only_agreement,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "category_agreement": Counter(row["category_agreement"] for row in rows),
                "presence_agreement": Counter(row["contact_presence_agreement"] for row in rows),
                "human_categories": Counter(row["human_machine_window_category"] for row in rows),
                "machine_categories": Counter(row["machine_closure_category"] for row in rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
