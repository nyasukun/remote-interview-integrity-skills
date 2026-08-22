#!/usr/bin/env python3
"""Render audio-only spectrogram sheets for a phoneme classification audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from video_integrity_analyzer.plosive_sync import decode_audio_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--half-window-s", type=float, default=0.25)
    parser.add_argument("--rows-per-sheet", type=int, default=6)
    parser.add_argument("--phoneme-class", default="p")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--max-frequency-hz", type=float, default=12_000.0)
    parser.add_argument(
        "--event-ids",
        nargs="*",
        help="Optional explicit runner ids; otherwise use analysis-eligible primary /p/ events.",
    )
    return parser.parse_args()


def load_candidate_times(path: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("status") != "measurable":
                continue
            candidates = json.loads(row.get("top3_candidates_json") or "[]")
            result[row["runner_event_id"]] = [
                float(item["absolute_time_s"]) for item in candidates
            ]
    return result


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.half_window_s) or args.half_window_s <= 0:
        raise SystemExit("--half-window-s must be positive")
    if args.rows_per_sheet <= 0:
        raise SystemExit("--rows-per-sheet must be positive")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive")
    if (
        not math.isfinite(args.max_frequency_hz)
        or not 0 < args.max_frequency_hz <= args.sample_rate / 2
    ):
        raise SystemExit("--max-frequency-hz must be in (0, sample_rate / 2]")
    payload = json.loads(args.events.read_text(encoding="utf-8"))["events"]
    selected_ids = set(args.event_ids or [])
    events = [
        {
            "runner_event_id": event["runner_event_id"],
            "audio_release_time_s": float(event["audio_release_time_s"]),
        }
        for event in payload
        if event.get("analysis_eligible")
        and (
            event["runner_event_id"] in selected_ids
            if selected_ids
            else event["selection"]["phoneme_class"] == args.phoneme_class
        )
    ]
    if selected_ids:
        by_id = {event["runner_event_id"]: event for event in events}
        missing = [event_id for event_id in args.event_ids if event_id not in by_id]
        if missing:
            raise RuntimeError(
                "Requested event IDs are missing or ineligible: " + ", ".join(missing)
            )
        events = [by_id[event_id] for event_id in args.event_ids]
    if not events:
        raise RuntimeError(
            f"No analysis-eligible /{args.phoneme_class}/ events selected"
        )
    candidate_times = load_candidate_times(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sheet_index in range(math.ceil(len(events) / args.rows_per_sheet)):
        chunk = events[
            sheet_index * args.rows_per_sheet : (sheet_index + 1) * args.rows_per_sheet
        ]
        fig, axes = plt.subplots(
            len(chunk),
            2,
            figsize=(15, 2.75 * len(chunk)),
            gridspec_kw={"width_ratios": [1.0, 2.15]},
            squeeze=False,
        )
        for row_index, event in enumerate(chunk):
            release = event["audio_release_time_s"]
            start = release - args.half_window_s
            end = release + args.half_window_s
            audio = decode_audio_window(
                args.video,
                start_s=start,
                end_s=end,
                sample_rate=args.sample_rate,
            )
            waveform = np.nan_to_num(audio.waveform, nan=0.0)
            times_ms = (
                start + np.arange(len(waveform), dtype=np.float64) / audio.sample_rate
                - release
            ) * 1000.0

            waveform_axis = axes[row_index, 0]
            waveform_axis.plot(times_ms, waveform, color="#111827", linewidth=0.45)
            waveform_axis.axvline(0.0, color="#16a34a", linewidth=1.2)
            waveform_axis.set_xlim(-args.half_window_s * 1000, args.half_window_s * 1000)
            waveform_axis.set_ylabel(event["runner_event_id"])
            waveform_axis.grid(alpha=0.12)

            frequencies, spec_times, spectrum = signal.spectrogram(
                waveform,
                fs=audio.sample_rate,
                window="hann",
                nperseg=720,
                noverlap=672,
                nfft=2048,
                mode="magnitude",
            )
            spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-7))
            spec_times_ms = (start + spec_times - release) * 1000.0
            frequency_mask = frequencies <= args.max_frequency_hz
            spectrum_axis = axes[row_index, 1]
            spectrum_axis.pcolormesh(
                spec_times_ms,
                frequencies[frequency_mask] / 1000.0,
                spectrum_db[frequency_mask],
                shading="auto",
                cmap="magma",
                vmin=-88,
                vmax=-28,
            )
            spectrum_axis.axvline(0.0, color="#22c55e", linewidth=1.2)
            for candidate in candidate_times.get(event["runner_event_id"], []):
                relative_ms = (candidate - release) * 1000.0
                if abs(relative_ms) > 0.5:
                    spectrum_axis.axvline(
                        relative_ms,
                        color="#60a5fa",
                        linewidth=0.75,
                        linestyle=":",
                        alpha=0.9,
                    )
            spectrum_axis.set_xlim(
                -args.half_window_s * 1000, args.half_window_s * 1000
            )
            spectrum_axis.set_ylim(0, args.max_frequency_hz / 1000.0)
            spectrum_axis.set_ylabel("kHz")
            spectrum_axis.set_title(
                f"{event['runner_event_id']}  release={release:.3f}s",
                loc="left",
                fontsize=10,
            )

        axes[-1, 0].set_xlabel("ms from fixed release")
        axes[-1, 1].set_xlabel("ms from fixed release")
        fig.suptitle(
            f"Audio-only /{args.phoneme_class}/ classification audit "
            "(green=fixed release, blue=other ranked onsets)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        output = args.output_dir / f"phoneme_classification_sheet_{sheet_index + 1:02d}.png"
        fig.savefig(output, dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
