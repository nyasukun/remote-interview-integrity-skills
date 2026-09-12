#!/usr/bin/env python3
"""Diagnose bundled synthetic audio and verify repeated-phrase codec timing.

Requires the AV toolkit's Python 3.12 environment with NumPy, SciPy, and PyAV.
Prints JSON to stdout, or writes --output outside the repository. The coarse ASR
anchors are search hints, not acoustic release references. Reused audio tests
sample timing only; it does not supply independent phoneme exemplars or accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AV_TOOLKIT = ROOT / "skills/interview-av-integrity/scripts/toolkit"
SOURCE = ROOT / "examples/synthetic-interview/deterministic_synthetic_interview_blind.mp4"
EXPECTED_SHA256 = "579bc0197f8bb03c0dc1d4250437d2fb7a77c074ec3829d7374718fe2055729c"
SAMPLE_RATE = 48_000
SEARCH_RADIUS_S = 0.030
MINIMUM_CORRELATION = 0.95
MAXIMUM_RESIDUAL_S = 0.003
BASELINE_START_S = 11.0
BASELINE_END_S = 15.5

# Labels and phone classes describe intended lexical contexts, not verified
# phoneme identities in the generated audio.
PROBES = (
    ("Python", "p", 11.38, 10.81, 12.62),
    ("FastApp", "p", 13.4225, 12.01, 13.96),
    ("API", "p", 14.413333, 13.63, 15.28),
    ("pa", "p", 18.10, 17.49, 18.66),
    ("bu", "b", 18.31, 17.81, 18.76),
    ("backup-pu", "p", 19.583333, 18.95, 20.02),
    ("response-po", "p", 21.04, 20.55, 21.48),
)
TARGET_WORD_WINDOWS_S = {
    "Python": (11.26, 12.22),
    "FastApp": (12.46, 13.56),
    "API": (14.08, 14.88),
    "pa": (17.94, 18.26),
    "bu": (18.26, 18.36),
    "backup-pu": (19.4, 19.62),
    "response-po": (21.0, 21.08),
}
NEIGHBOUR_WORD_WINDOWS_S = {
    "Python": ((10.98, 11.26), (12.22, 12.46)),
    "FastApp": ((12.22, 12.46), (13.56, 14.08)),
    "API": ((13.56, 14.08), (14.88, 15.04)),
    "pa": ((17.76, 17.94), (18.26, 18.36)),
    "bu": ((17.94, 18.26), (18.36, 18.44)),
    "backup-pu": ((19.10, 19.40), (19.62, 19.74)),
    "response-po": ((20.88, 21.00), (21.08, 21.14)),
}
REPEATS = (
    ("audio_delayed_four_frames", 35.0, 41.5, 4 / 24),
    ("audio_advanced_five_frames", 42.5, 49.0, -5 / 24),
    ("audio_unshifted", 50.0, 56.5, 0.0),
)


def source_sha256() -> str:
    digest = hashlib.sha256()
    with SOURCE.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_coverage(window: Any, label: str, failures: list[str]) -> dict[str, Any]:
    import numpy as np

    covered = np.asarray(window.coverage_mask, dtype=bool)
    finite = np.isfinite(window.waveform)
    complete = bool(covered.size and np.all(covered) and np.all(finite))
    if not complete:
        failures.append(f"incomplete_audio_coverage:{label}")
    return {
        "start_s": window.start_s,
        "end_s": window.end_s,
        "sample_count": len(window.waveform),
        "coverage_fraction": window.coverage_fraction,
        "complete": complete,
        "warnings": list(window.warnings),
    }


def match_repeated_audio(reference: Any, target: Any) -> dict[str, Any]:
    """Find signed, zero-mean normalized correlation at each valid sample lag."""
    import numpy as np
    from scipy.signal import correlate

    baseline = np.asarray(reference.waveform, dtype=np.float64)
    search = np.asarray(target.waveform, dtype=np.float64)
    count = len(baseline)
    if not count or len(search) < count:
        raise ValueError("correlation windows do not contain enough samples")
    centered = baseline - np.mean(baseline)
    reference_energy = float(np.dot(centered, centered))
    if reference_energy <= 0:
        raise ValueError("baseline has no varying audio signal")

    # Each target position has its own mean and energy. This is normalized
    # correlation, not a maximum of unnormalized amplitudes or absolute values.
    sums = np.concatenate(([0.0], np.cumsum(search)))
    squares = np.concatenate(([0.0], np.cumsum(search * search)))
    local_sums = sums[count:] - sums[:-count]
    local_energy = squares[count:] - squares[:-count] - local_sums**2 / count
    numerator = correlate(search, centered, mode="valid", method="fft")
    numerator -= local_sums * float(np.sum(centered)) / count
    denominator = np.sqrt(np.maximum(local_energy, 0.0) * reference_energy)
    correlations = np.full(numerator.shape, -np.inf)
    np.divide(numerator, denominator, out=correlations, where=denominator > 0)
    if not np.any(np.isfinite(correlations)):
        raise ValueError("target has no varying audio signal")
    peak_index = int(np.argmax(correlations))
    peak = float(np.clip(correlations[peak_index], -1.0, 1.0))
    match_start_s = target.start_s + peak_index / SAMPLE_RATE
    return {
        "peak_correlation": peak,
        "matched_target_start_s": match_start_s,
        "measured_shift_s": match_start_s - reference.start_s,
        "peak_index_samples": peak_index,
        "sample_resolution_s": 1 / SAMPLE_RATE,
    }


def evaluate(report: dict[str, Any]) -> None:
    import av
    import numpy as np
    import scipy

    # Prefer this checkout even when another toolkit has been installed.
    sys.path.insert(0, str(AV_TOOLKIT))
    from video_integrity_analyzer.plosive_sync import (
        AcousticReleaseConfig,
        decode_audio_window,
        estimate_acoustic_release,
    )

    failures = report["failures"]
    report["runtime"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyav": av.__version__,
    }
    config = AcousticReleaseConfig.conservative()
    report["acoustic_config"] = asdict(config)
    report["diagnostic_probes"] = []
    for label, phone_class, anchor, start, end in PROBES:
        window = decode_audio_window(
            SOURCE, start_s=start, end_s=end, sample_rate=SAMPLE_RATE
        )
        coverage = check_coverage(window, label, failures)
        estimate = estimate_acoustic_release(
            window.waveform,
            window.sample_rate,
            window_start_s=window.start_s,
            anchor_time_s=anchor,
            phone_class=phone_class,
            coverage_mask=window.coverage_mask,
            config=config,
            target_window_s=TARGET_WORD_WINDOWS_S[label],
            previous_window_s=NEIGHBOUR_WORD_WINDOWS_S[label][0],
            next_window_s=NEIGHBOUR_WORD_WINDOWS_S[label][1],
        )
        report["diagnostic_probes"].append({
            "context_label": label,
            "anchor_status": "coarse_asr_search_hint_not_release_truth",
            "phone_class_status": "intended_context_not_verified_phoneme",
            "requires_audio_realization_review": True,
            "target_word_window_s": list(TARGET_WORD_WINDOWS_S[label]),
            "neighbour_word_windows_s": NEIGHBOUR_WORD_WINDOWS_S[label],
            "window": coverage,
            "status": "accepted" if estimate.measurable else "rejected",
            "estimate": estimate.as_dict(),
        })

    baseline = decode_audio_window(
        SOURCE,
        start_s=BASELINE_START_S,
        end_s=BASELINE_END_S,
        sample_rate=SAMPLE_RATE,
    )
    baseline_coverage = check_coverage(baseline, "repeated_audio_baseline", failures)
    report["repeated_audio_timing"] = {
        "interpretation": "same_phrase_reuse_not_independent_exemplars",
        "source_copy_interval_s": [10.0, 16.5],
        "baseline_window": baseline_coverage,
        "search_radius_s": SEARCH_RADIUS_S,
        "minimum_peak_correlation": MINIMUM_CORRELATION,
        "maximum_absolute_residual_s": MAXIMUM_RESIDUAL_S,
        "copies": [],
    }
    for label, output_start, output_end, audio_offset in REPEATS:
        expected_shift = output_start - 10.0 + audio_offset
        target = decode_audio_window(
            SOURCE,
            start_s=BASELINE_START_S + expected_shift - SEARCH_RADIUS_S,
            end_s=BASELINE_END_S + expected_shift + SEARCH_RADIUS_S,
            sample_rate=SAMPLE_RATE,
        )
        coverage = check_coverage(target, label, failures)
        result = {
            "label": label,
            "output_copy_interval_s": [output_start, output_end],
            "audio_offset_from_copy_s": audio_offset,
            "expected_shift_s": expected_shift,
            "target_window": coverage,
            "passed": False,
        }
        if baseline_coverage["complete"] and coverage["complete"]:
            result.update(match_repeated_audio(baseline, target))
            residual = result["measured_shift_s"] - expected_shift
            result["residual_s"] = residual
            correlation_passed = result["peak_correlation"] >= MINIMUM_CORRELATION
            timing_passed = abs(residual) <= MAXIMUM_RESIDUAL_S
            result["passed"] = correlation_passed and timing_passed
            if not correlation_passed:
                failures.append(f"repeated_audio_correlation_below_threshold:{label}")
            if not timing_passed:
                failures.append(f"repeated_audio_timing_residual_exceeds_tolerance:{label}")
        report["repeated_audio_timing"]["copies"].append(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        help="write JSON to this path outside the repository instead of stdout",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve() if args.output else None
    if output is not None and output.is_relative_to(ROOT):
        parser.error("--output must be outside the repository")

    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_role": "automated_acoustic_candidate_diagnostics",
        "fixture": str(SOURCE.relative_to(ROOT)),
        "expected_sha256": EXPECTED_SHA256,
        "limitations": [
            "ASR anchors are deliberately coarse and are not release-time truth.",
            "Diagnostic acceptance is not per-phoneme validation or an accuracy result.",
            "Repeated copies reuse one phrase and are not independent exemplars.",
            "Correlation verifies decoded audio sample timing, not visual synchrony.",
        ],
        "failures": [],
    }
    try:
        actual_hash = source_sha256()
        report["actual_sha256"] = actual_hash
        report["hash_verified"] = actual_hash == EXPECTED_SHA256
        if not report["hash_verified"]:
            report["failures"].append("fixture_sha256_mismatch")
        else:
            # Keep imports that may create caches inside this temporary scope.
            with tempfile.TemporaryDirectory(prefix="synthetic-acoustics-") as temporary:
                os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
                os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
                os.environ["MPLBACKEND"] = "Agg"
                os.environ["XDG_CACHE_HOME"] = str(Path(temporary) / "cache")
                evaluate(report)
    except Exception as error:
        report["failures"].append(f"evaluation_error:{type(error).__name__}:{error}")
    report["passed"] = not report["failures"]
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(serialized)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
