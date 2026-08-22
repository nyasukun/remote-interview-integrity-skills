from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EpochEstimate:
    """One independent-ish synchronization epoch.

    Multiple phoneme events from the same uninterrupted participant stream share
    a jitter buffer and are therefore summarized before group comparison.
    """

    speaker: str
    group: str
    epoch_id: str
    event_count: int
    median_lag_ms: float
    mad_lag_ms: float
    fraction_over_margin: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GroupComparison:
    candidate_epoch_count: int
    control_epoch_count: int
    candidate_event_count: int
    control_event_count: int
    candidate_median_ms: float | None
    control_median_ms: float | None
    median_difference_ms: float | None
    median_difference_frames: float | None
    bootstrap_ci95_ms: tuple[float, float] | None
    probability_candidate_greater: float | None
    cliffs_delta: float | None
    exploratory_one_sided_permutation_p: float | None
    candidate_fraction_epochs_over_margin: float | None
    control_fraction_epochs_over_margin: float | None
    margin_ms: float
    cautions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    return result[np.isfinite(result)]


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def summarize_epochs(
    events: Iterable[Mapping[str, object]],
    *,
    margin_ms: float = 1000.0 / 24.0 * 2.0,
) -> list[EpochEstimate]:
    """Collapse measurable event lags to epoch medians.

    Required fields are ``speaker``, ``group``, ``epoch_id``, and
    ``release_lag_ms``. Events with an explicit false ``measurable`` value or a
    non-finite lag stay in the audit table but do not enter the comparison.
    """
    if not math.isfinite(margin_ms) or margin_ms < 0:
        raise ValueError("margin_ms must be a finite non-negative number")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for event in events:
        if event.get("measurable", True) is False:
            continue
        try:
            lag = float(event["release_lag_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(lag):
            continue
        speaker = str(event["speaker"])
        group = str(event["group"])
        epoch_id = str(event["epoch_id"])
        grouped.setdefault((speaker, group, epoch_id), []).append(lag)

    estimates: list[EpochEstimate] = []
    for (speaker, group, epoch_id), lags in sorted(grouped.items()):
        values = np.asarray(lags, dtype=np.float64)
        estimates.append(
            EpochEstimate(
                speaker=speaker,
                group=group,
                epoch_id=epoch_id,
                event_count=int(values.size),
                median_lag_ms=float(np.median(values)),
                mad_lag_ms=_mad(values),
                fraction_over_margin=float(np.mean(values > margin_ms)),
            )
        )
    return estimates


def _probability_greater(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    comparisons = left[:, None] - right[None, :]
    wins = float(np.sum(comparisons > 0))
    ties = float(np.sum(comparisons == 0))
    total = float(comparisons.size)
    probability = (wins + 0.5 * ties) / total
    return probability, 2.0 * probability - 1.0


def _bootstrap_median_difference(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[tuple[float, float], float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        candidate_draw = rng.choice(candidate, size=candidate.size, replace=True)
        control_draw = rng.choice(control, size=control.size, replace=True)
        differences[index] = np.median(candidate_draw) - np.median(control_draw)
    low, high = np.quantile(differences, (0.025, 0.975))
    probability = float(np.mean(differences > 0.0))
    return (float(low), float(high)), probability


def _one_sided_label_permutation_p(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    exact_limit: int,
    iterations: int,
    seed: int,
) -> float:
    """Exploratory label permutation on epoch medians.

    This is deliberately labelled exploratory because participant identity and
    network path are not exchangeable within a recorded call.
    """
    combined = np.concatenate((candidate, control))
    candidate_size = candidate.size
    observed = float(np.median(candidate) - np.median(control))
    combination_count = math.comb(combined.size, candidate_size)
    greater_or_equal = 0
    total = 0
    if combination_count <= exact_limit:
        selections: Iterable[Sequence[int]] = itertools.combinations(
            range(combined.size), candidate_size
        )
    else:
        rng = np.random.default_rng(seed)
        selections = (
            tuple(rng.choice(combined.size, size=candidate_size, replace=False))
            for _ in range(iterations)
        )
    all_indices = np.arange(combined.size)
    for selection in selections:
        candidate_indices = np.asarray(selection, dtype=np.int64)
        control_mask = np.ones(combined.size, dtype=bool)
        control_mask[candidate_indices] = False
        statistic = float(
            np.median(combined[candidate_indices]) - np.median(combined[control_mask])
        )
        greater_or_equal += int(statistic >= observed - 1e-12)
        total += 1
    return float((greater_or_equal + 1) / (total + 1))


def compare_candidate_to_controls(
    epochs: Sequence[EpochEstimate],
    *,
    candidate_group: str = "candidate",
    control_group: str = "control",
    fps: float = 24.0,
    margin_ms: float = 1000.0 / 24.0 * 2.0,
    bootstrap_iterations: int = 20_000,
    permutation_iterations: int = 100_000,
    seed: int = 1729,
) -> GroupComparison:
    """Compare epoch medians; positive values mean video lags audio more.

    The function quantifies this recording only. It must not be interpreted as
    a population-level biometric or identity test.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if bootstrap_iterations <= 0 or permutation_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    candidate_epochs = [epoch for epoch in epochs if epoch.group == candidate_group]
    control_epochs = [epoch for epoch in epochs if epoch.group == control_group]
    candidate = _finite(epoch.median_lag_ms for epoch in candidate_epochs)
    control = _finite(epoch.median_lag_ms for epoch in control_epochs)
    cautions = (
        "Epochs, not phoneme tokens, are the comparison unit to reduce pseudoreplication.",
        "Speaker identity is confounded with each participant's device, network path, and Meet jitter buffer.",
        "The permutation p-value is exploratory because participant streams are not randomized or exchangeable.",
        "A difference can establish a stream-specific A/V anomaly, not its cause or the speaker's identity.",
    )
    if candidate.size == 0 or control.size == 0:
        return GroupComparison(
            candidate_epoch_count=int(candidate.size),
            control_epoch_count=int(control.size),
            candidate_event_count=sum(epoch.event_count for epoch in candidate_epochs),
            control_event_count=sum(epoch.event_count for epoch in control_epochs),
            candidate_median_ms=None,
            control_median_ms=None,
            median_difference_ms=None,
            median_difference_frames=None,
            bootstrap_ci95_ms=None,
            probability_candidate_greater=None,
            cliffs_delta=None,
            exploratory_one_sided_permutation_p=None,
            candidate_fraction_epochs_over_margin=None,
            control_fraction_epochs_over_margin=None,
            margin_ms=float(margin_ms),
            cautions=cautions,
        )

    candidate_median = float(np.median(candidate))
    control_median = float(np.median(control))
    difference = candidate_median - control_median
    ci95, bootstrap_probability = _bootstrap_median_difference(
        candidate,
        control,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    pair_probability, cliffs_delta = _probability_greater(candidate, control)
    p_value = _one_sided_label_permutation_p(
        candidate,
        control,
        exact_limit=250_000,
        iterations=permutation_iterations,
        seed=seed + 1,
    )
    # The pairwise probability describes observed epoch ordering, while the
    # bootstrap probability describes uncertainty under epoch resampling. Save
    # the more conservative of the two under a single, explicit field.
    probability_greater = min(pair_probability, bootstrap_probability)
    return GroupComparison(
        candidate_epoch_count=int(candidate.size),
        control_epoch_count=int(control.size),
        candidate_event_count=sum(epoch.event_count for epoch in candidate_epochs),
        control_event_count=sum(epoch.event_count for epoch in control_epochs),
        candidate_median_ms=candidate_median,
        control_median_ms=control_median,
        median_difference_ms=difference,
        median_difference_frames=difference * fps / 1000.0,
        bootstrap_ci95_ms=ci95,
        probability_candidate_greater=probability_greater,
        cliffs_delta=cliffs_delta,
        exploratory_one_sided_permutation_p=p_value,
        candidate_fraction_epochs_over_margin=float(np.mean(candidate > margin_ms)),
        control_fraction_epochs_over_margin=float(np.mean(control > margin_ms)),
        margin_ms=float(margin_ms),
        cautions=cautions,
    )
