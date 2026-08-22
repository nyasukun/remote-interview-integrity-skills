#!/usr/bin/env python3
"""Build a blinded multi-reviewer contact consensus, then restore metadata."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from build_direction_closure_blind_sheets import normalized_review_value


CONTACT_VALUES = ("yes", "no", "ambiguous")
CONTACT_VALUE_ALIASES = {
    "yes": "yes",
    "contact": "yes",
    "no": "no",
    "no_contact": "no",
    "ambiguous": "ambiguous",
}


def read_keyed_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {str(row.get("blind_id", "")).strip(): row for row in rows}
    if "" in keyed or not keyed:
        raise RuntimeError(f"{path} must contain nonempty blind_id values")
    if len(keyed) != len(rows):
        raise RuntimeError(f"Duplicate blind_id in {path}")
    return keyed


def validate_contact(value: str, *, source: str, blind_id: str) -> str:
    value = value.strip().lower()
    normalized = CONTACT_VALUE_ALIASES.get(value)
    if normalized is None:
        raise RuntimeError(
            f"Unexpected contact value {value!r} from {source} for {blind_id}"
        )
    return normalized


def strict_contact_majority(ratings: Sequence[str]) -> str:
    """Return a strict yes/no majority; ties and all other cases are ambiguous."""

    if len(ratings) < 2:
        raise ValueError("Contact consensus needs at least two ratings")
    votes = Counter(ratings)
    threshold = len(ratings) // 2 + 1
    if votes["yes"] >= threshold:
        return "yes"
    if votes["no"] >= threshold:
        return "no"
    return "ambiguous"


def cohen_kappa(
    values_a: list[str], values_b: list[str]
) -> tuple[float, float | None]:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Cohen kappa needs equal non-empty rating vectors")
    total = len(values_a)
    observed = sum(a == b for a, b in zip(values_a, values_b, strict=True)) / total
    counts_a = Counter(values_a)
    counts_b = Counter(values_b)
    expected = sum(
        counts_a[category] / total * counts_b[category] / total
        for category in CONTACT_VALUES
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return observed, kappa


def fleiss_kappa(
    ratings: Sequence[Sequence[str]],
) -> tuple[float, float | None]:
    if not ratings:
        raise ValueError("Fleiss kappa needs ratings")
    rater_count = len(ratings[0])
    if rater_count < 2 or any(len(item) != rater_count for item in ratings):
        raise ValueError("Fleiss kappa needs equal rating vectors from 2+ raters")
    category_totals: Counter[str] = Counter()
    per_item_agreement: list[float] = []
    for item in ratings:
        counts = Counter(item)
        category_totals.update(counts)
        per_item_agreement.append(
            (sum(counts[category] ** 2 for category in CONTACT_VALUES) - rater_count)
            / (rater_count * (rater_count - 1))
        )
    observed = sum(per_item_agreement) / len(per_item_agreement)
    total_ratings = len(ratings) * rater_count
    proportions = {
        category: category_totals[category] / total_ratings
        for category in CONTACT_VALUES
    }
    expected = sum(proportion**2 for proportion in proportions.values())
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return observed, kappa


def reviewer_paths_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[Path]:
    repeated = list(args.reviewer or [])
    legacy = [
        path
        for path in (
            args.reviewer1_blind,
            args.reviewer2_blind,
            args.reviewer3_blind,
        )
        if path is not None
    ]
    if repeated and legacy:
        parser.error("Use either repeated --reviewer or the legacy reviewer flags")
    paths = repeated or legacy
    if len(paths) < 2:
        parser.error("At least two blinded reviewer CSV files are required")
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        parser.error("Reviewer paths must be unique")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewer",
        type=Path,
        action="append",
        help="Blinded reviewer CSV; repeat once per reviewer (minimum two).",
    )
    parser.add_argument("--reviewer1-blind", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reviewer2-blind", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reviewer3-blind", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reviewer_paths = reviewer_paths_from_args(parser, args)
    reviewers = [read_keyed_csv(path) for path in reviewer_paths]
    rater_count = len(reviewers)
    blind_ids = sorted(reviewers[0])
    expected_ids = set(blind_ids)
    for reviewer_index, reviewer in enumerate(reviewers, start=1):
        if set(reviewer) != expected_ids:
            raise RuntimeError(
                f"Reviewer {reviewer_index} does not contain the same blind IDs"
            )

    # This phase intentionally uses no event, speaker, or group metadata.
    blind_consensus: list[dict[str, object]] = []
    rating_vectors: list[tuple[str, ...]] = []
    for blind_id in blind_ids:
        ratings = tuple(
            validate_contact(
                normalized_review_value(reviewer[blind_id], "contact"),
                source=f"reviewer{reviewer_index}",
                blind_id=blind_id,
            )
            for reviewer_index, reviewer in enumerate(reviewers, start=1)
        )
        rating_vectors.append(ratings)
        votes = Counter(ratings)
        majority = strict_contact_majority(ratings)
        unanimous = len(votes) == 1
        row: dict[str, object] = {
            "blind_id": blind_id,
            "rater_count": rater_count,
        }
        row.update(
            {
                f"reviewer{reviewer_index}_contact": contact
                for reviewer_index, contact in enumerate(ratings, start=1)
            }
        )
        row.update(
            {
                "yes_votes": votes["yes"],
                "no_votes": votes["no"],
                "ambiguous_votes": votes["ambiguous"],
                "contact_majority_threshold": rater_count // 2 + 1,
                "contact_majority": majority,
                # Preserve the historical field for legacy three-rater consumers.
                "contact_majority_2of3": majority if rater_count == 3 else "",
                "contact_unanimous": str(unanimous).lower(),
                "contact_unanimous_category": ratings[0] if unanimous else "none",
            }
        )
        row.update(
            {
                f"reviewer{reviewer_index}_timing": normalized_review_value(
                    reviewer[blind_id], "timing_relative_audio", required=False
                )
                for reviewer_index, reviewer in enumerate(reviewers, start=1)
            }
        )
        row.update(
            {
                f"reviewer{reviewer_index}_machine_window_category": normalized_review_value(
                    reviewer[blind_id], "machine_window_category", required=False
                )
                for reviewer_index, reviewer in enumerate(reviewers, start=1)
            }
        )
        blind_consensus.append(row)

    # Restore the unblinded key only after every consensus decision is fixed.
    metadata = read_keyed_csv(args.metadata)
    if set(metadata) != set(blind_ids):
        raise RuntimeError("Metadata must contain the same blind IDs")
    output_rows: list[dict[str, object]] = []
    for row in blind_consensus:
        event = metadata[str(row["blind_id"])]
        output_rows.append(
            {
                **row,
                "blind_order": event["blind_order"],
                "runner_event_id": event["runner_event_id"],
                "speaker": event["speaker"],
                "group": event["group"],
                "epoch_id": event["epoch_id"],
                "kana": event["kana"],
                "token_text": event["token_text"],
                "audio_release_time_s": event["audio_release_time_s"],
                "machine_closure_category": event["machine_closure_category"],
                "primary_followup_first_contact_offset_ms": event.get(
                    "human_followup_first_contact_offset_ms",
                    event.get("human_first_contact_through_plus500_ms", ""),
                ),
                "primary_release_after_contact": event.get(
                    "human_release_after_contact", ""
                ),
                "primary_release_after_contact_offset_ms": event.get(
                    "human_release_after_contact_offset_ms", ""
                ),
                "primary_followup_confidence": event.get(
                    "human_followup_confidence", ""
                ),
                "primary_followup_note": event.get("human_followup_note", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    pairwise: dict[str, dict[str, float | None]] = {}
    for left, right in itertools.combinations(range(rater_count), 2):
        observed, kappa = cohen_kappa(
            [ratings[left] for ratings in rating_vectors],
            [ratings[right] for ratings in rating_vectors],
        )
        pairwise[f"reviewer{left + 1}_reviewer{right + 1}"] = {
            "observed_agreement": observed,
            "cohen_kappa": kappa,
        }
    fleiss_observed, fleiss_value = fleiss_kappa(rating_vectors)
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "rater_count": rater_count,
                "majority_rule": "strict_majority_of_all_raters",
                "contact_majority_threshold": rater_count // 2 + 1,
                "majority": Counter(
                    str(row["contact_majority"]) for row in output_rows
                ),
                "unanimous": Counter(
                    str(row["contact_unanimous_category"])
                    for row in output_rows
                ),
                "group_majority": {
                    group: Counter(
                        str(row["contact_majority"])
                        for row in output_rows
                        if row["group"] == group
                    )
                    for group in sorted({str(row["group"]) for row in output_rows})
                },
                "pairwise": pairwise,
                "fleiss_observed_agreement": fleiss_observed,
                "fleiss_kappa": fleiss_value,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
