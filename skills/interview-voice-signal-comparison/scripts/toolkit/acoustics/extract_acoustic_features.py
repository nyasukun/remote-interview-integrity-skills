#!/usr/bin/env python3
"""Extract local, non-biometric descriptive acoustic artifacts from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Re-export the existing manifest API for standalone callers.
    from input_manifest import (  # type: ignore[import-not-found]
        COLOR_PATTERN,
        DEFAULT_COLORS,
        ID_PATTERN,
        ClipSpec,
        GroupSpec,
        InputSpec,
        load_manifest,
    )
    from model import (  # type: ignore[import-not-found]
        ACOUSTIC_SCHEMA_VERSION,
        BOUNDARY_TEXT,
        METRICS,
        implementation_provenance,
        json_ready,
        scalar_metrics,
    )
    from signal_features import (  # type: ignore[import-not-found]
        DecodedAudio,
        FeatureBundle,
        FeatureConfig,
        analyze_signal,
        decode_audio,
        sha256_file,
    )
else:
    # Re-export the existing manifest API for standalone callers.
    from .input_manifest import (
        COLOR_PATTERN,
        DEFAULT_COLORS,
        ID_PATTERN,
        ClipSpec,
        GroupSpec,
        InputSpec,
        load_manifest,
    )
    from .model import (
        ACOUSTIC_SCHEMA_VERSION,
        BOUNDARY_TEXT,
        METRICS,
        implementation_provenance,
        json_ready,
        scalar_metrics,
    )
    from .signal_features import (
        DecodedAudio,
        FeatureBundle,
        FeatureConfig,
        analyze_signal,
        decode_audio,
        sha256_file,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value)).lower()
    if isinstance(value, (np.floating, float)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_group_summaries(
    rows: Sequence[tuple[ClipSpec, FeatureBundle]],
    groups: Sequence[GroupSpec],
) -> list[dict[str, Any]]:
    group_lookup = {group.group_id: group for group in groups}
    grouped: dict[str, list[tuple[ClipSpec, FeatureBundle]]] = {}
    for row in rows:
        grouped.setdefault(row[0].group_id, []).append(row)
    summaries: list[dict[str, Any]] = []
    for group in groups:
        group_rows = grouped[group.group_id]
        scalar_rows = [scalar_metrics(bundle.summary) for _, bundle in group_rows]
        metrics: dict[str, dict[str, float | int | None]] = {}
        for definition in METRICS:
            finite = [
                row[definition.metric_id]
                for row in scalar_rows
                if row[definition.metric_id] is not None
            ]
            metrics[definition.metric_id] = {
                "value_count": len(finite),
                "min": min(finite) if finite else None,
                "max": max(finite) if finite else None,
                "median": float(np.median(finite)) if finite else None,
            }
        summaries.append(
            {
                "group_id": group.group_id,
                "group_label": group.group_label,
                "display_color_hex": group_lookup[group.group_id].display_color_hex,
                "clip_count": len(group_rows),
                "clip_ids": [spec.clip_id for spec, _ in group_rows],
                "aggregation_unit": "unweighted per-clip scalar summaries",
                "metrics": metrics,
            }
        )
    return summaries


def _write_frame_csv(path: Path, rows: Sequence[tuple[ClipSpec, FeatureBundle]]) -> None:
    feature_names = list(rows[0][1].frame_columns)
    fields = [
        "clip_id",
        "group_id",
        "frame_index",
        "clip_time_s",
        "source_time_s",
        "analysis_center_sample_index",
        *[name for name in feature_names if name != "time_s"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec, bundle in rows:
            for index, clip_time_s in enumerate(bundle.frame_times_s):
                record = {
                    "clip_id": spec.clip_id,
                    "group_id": spec.group_id,
                    "frame_index": index,
                    "clip_time_s": float(clip_time_s),
                    "source_time_s": float(spec.start_s + clip_time_s),
                    "analysis_center_sample_index": int(
                        round(float(clip_time_s) * bundle.sample_rate)
                    ),
                }
                for name in feature_names:
                    if name != "time_s":
                        record[name] = _csv_value(bundle.frame_columns[name][index])
                writer.writerow(record)


def _write_clip_csv(
    path: Path,
    rows: Sequence[tuple[ClipSpec, FeatureBundle]],
    group_lookup: Mapping[str, GroupSpec],
) -> None:
    fields = [
        "clip_id",
        "group_id",
        "group_label",
        "display_color_hex",
        "clip_label",
        *[definition.metric_id for definition in METRICS],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec, bundle in rows:
            group = group_lookup[spec.group_id]
            writer.writerow(
                {
                    "clip_id": spec.clip_id,
                    "group_id": spec.group_id,
                    "group_label": group.group_label,
                    "display_color_hex": group.display_color_hex,
                    "clip_label": spec.clip_label,
                    **{key: _csv_value(value) for key, value in scalar_metrics(bundle.summary).items()},
                }
            )


def _write_group_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "group_id",
        "group_label",
        "display_color_hex",
        "clip_count",
        "clip_ids",
        "metric_id",
        "unit",
        "value_count",
        "min",
        "max",
        "median",
    ]
    definitions = {definition.metric_id: definition for definition in METRICS}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for metric_id, values in summary["metrics"].items():
                writer.writerow(
                    {
                        "group_id": summary["group_id"],
                        "group_label": summary["group_label"],
                        "display_color_hex": summary["display_color_hex"],
                        "clip_count": summary["clip_count"],
                        "clip_ids": "|".join(summary["clip_ids"]),
                        "metric_id": metric_id,
                        "unit": definitions[metric_id].unit,
                        "value_count": values["value_count"],
                        "min": _csv_value(values["min"]),
                        "max": _csv_value(values["max"]),
                        "median": _csv_value(values["median"]),
                    }
                )


def _plot_clip(
    output: Path,
    spec: ClipSpec,
    bundle: FeatureBundle,
    color: str,
    mel_limits: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), constrained_layout=True)
    duration_s = len(bundle.samples) / bundle.sample_rate
    time_s = np.arange(len(bundle.samples)) / bundle.sample_rate
    mono = np.mean(bundle.samples, axis=1)
    peak = max(float(np.max(np.abs(mono))), 1e-9)
    stride = max(1, len(mono) // 20_000)
    axes[0].plot(time_s[::stride], mono[::stride] / peak, color=color, linewidth=0.6)
    axes[0].set(
        xlim=(0, duration_s),
        ylim=(-1.05, 1.05),
        ylabel="normalized",
        title=f"{spec.clip_id} | display-normalized waveform (not absolute loudness)",
    )
    image = axes[1].imshow(
        bundle.log_mel_power_db.T,
        origin="lower",
        aspect="auto",
        extent=(
            float(bundle.frame_times_s[0]),
            float(bundle.frame_times_s[-1]),
            float(bundle.mel_frequencies_hz[0]),
            float(bundle.mel_frequencies_hz[-1]),
        ),
        cmap="magma",
        vmin=mel_limits[0],
        vmax=mel_limits[1],
    )
    fig.colorbar(image, ax=axes[1], label="log mel power (dB, shared scale)")
    axes[1].set(ylabel="Hz", title="Log-Mel spectrum")
    axes[2].plot(bundle.frame_times_s, bundle.frame_columns["f0_hz"], color=color)
    axes[2].set(ylim=(70, 310), ylabel="F0 (Hz)", title="Voiced F0 and periodicity")
    periodicity_axis = axes[2].twinx()
    periodicity_axis.plot(
        bundle.frame_times_s,
        bundle.frame_columns["periodicity"],
        color="#777777",
        linewidth=0.65,
    )
    periodicity_axis.set(ylim=(0, 1.02), ylabel="periodicity")
    bottom = np.zeros(len(bundle.frame_times_s))
    band_colors = ("#56B4E9", "#009E73", "#E69F00", "#999999")
    band_names = [
        name.removeprefix("band_fraction_")
        for name in bundle.frame_columns
        if name.startswith("band_fraction_")
    ]
    for index, name in enumerate(band_names):
        values = bundle.frame_columns[f"band_fraction_{name}"]
        axes[3].fill_between(
            bundle.frame_times_s,
            bottom,
            bottom + values,
            color=band_colors[index % len(band_colors)],
            label=name,
            linewidth=0,
        )
        bottom += np.nan_to_num(values)
    axes[3].set(ylim=(0, 1), ylabel="power fraction", title="Spectral bands")
    axes[3].legend(loc="upper right", ncol=2, fontsize="small")
    correlation = bundle.frame_columns["stereo_correlation"]
    if np.any(np.isfinite(correlation)):
        axes[4].plot(
            bundle.frame_times_s,
            bundle.frame_columns["stereo_balance_db"],
            color="#6A3D9A",
        )
        axes[4].set(ylabel="balance dB", title="Stereo channel diagnostics")
        correlation_axis = axes[4].twinx()
        correlation_axis.plot(bundle.frame_times_s, correlation, color="#1B9E77")
        correlation_axis.set(ylim=(-1.05, 1.05), ylabel="correlation")
    else:
        axes[4].text(0.5, 0.5, "Mono: stereo diagnostics unavailable", ha="center")
        axes[4].set_axis_off()
    axes[-1].set_xlabel("clip time (s)")
    fig.text(0.5, 0.004, BOUNDARY_TEXT, ha="center", fontsize=8, color="#444444")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_overview(
    output: Path,
    rows: Sequence[tuple[ClipSpec, FeatureBundle]],
    group_lookup: Mapping[str, GroupSpec],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for spec, bundle in rows:
        color = group_lookup[spec.group_id].display_color_hex
        label = f"{spec.clip_id} [{spec.group_id}]"
        f0 = bundle.frame_columns["f0_hz"]
        f0 = f0[np.isfinite(f0)]
        if f0.size:
            axes[0, 0].hist(
                f0,
                bins=np.linspace(80, 300, 45),
                histtype="step",
                density=True,
                color=color,
                label=label,
            )
        active = bundle.frame_columns["active"].astype(bool)
        axes[0, 1].hist(
            bundle.frame_columns["periodicity"][active],
            bins=np.linspace(0, 1, 41),
            histtype="step",
            density=True,
            color=color,
            label=label,
        )
    axes[0, 0].set(title="Voiced F0 distributions", xlabel="Hz", ylabel="density")
    axes[0, 1].set(title="Active-frame periodicity", xlabel="peak", ylabel="density")
    axes[0, 0].legend(fontsize="x-small")
    axes[0, 1].legend(fontsize="x-small")
    band_ids = [
        definition.metric_id
        for definition in METRICS
        if definition.metric_id.startswith("mean_active_band_fraction_")
    ]
    positions = np.arange(len(band_ids))
    width = 0.8 / len(rows)
    for index, (spec, bundle) in enumerate(rows):
        scalars = scalar_metrics(bundle.summary)
        axes[1, 0].bar(
            positions - 0.4 + width / 2 + index * width,
            [scalars[metric_id] or 0.0 for metric_id in band_ids],
            width=width,
            color=group_lookup[spec.group_id].display_color_hex,
            label=f"{spec.clip_id} [{spec.group_id}]",
        )
    axes[1, 0].set_xticks(
        positions,
        [item.replace("mean_active_band_fraction_", "") for item in band_ids],
        rotation=20,
        ha="right",
    )
    axes[1, 0].set(title="Mean active spectral fractions", ylabel="fraction")
    axes[1, 0].legend(fontsize="x-small")
    for spec, bundle in rows:
        scalars = scalar_metrics(bundle.summary)
        balance = scalars["stereo_balance_median_db"]
        correlation = scalars["stereo_correlation_median"]
        if balance is not None and correlation is not None:
            axes[1, 1].scatter(
                [balance],
                [correlation],
                color=group_lookup[spec.group_id].display_color_hex,
                label=f"{spec.clip_id} [{spec.group_id}]",
            )
    axes[1, 1].set(
        title="Stereo balance and correlation",
        xlabel="balance dB",
        ylabel="correlation",
        ylim=(-1.05, 1.05),
    )
    axes[1, 1].legend(fontsize="x-small")
    fig.suptitle("Descriptive caller-labeled group signal observations")
    fig.text(0.5, 0.004, BOUNDARY_TEXT, ha="center", fontsize=8, color="#444444")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in names}


def extract(manifest_path: Path, output_dir: Path, sample_rate: int = 48_000) -> dict[str, Any]:
    spec = load_manifest(manifest_path)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = FeatureConfig(sample_rate=sample_rate)
    config.validate()
    rows: list[tuple[ClipSpec, FeatureBundle]] = []
    decoded_rows: list[DecodedAudio] = []
    for clip in spec.clips:
        decoded = decode_audio(
            clip.path,
            sample_rate=sample_rate,
            start_s=clip.start_s,
            end_s=clip.end_s,
        )
        decoded_rows.append(decoded)
        rows.append(
            (
                clip,
                analyze_signal(
                    decoded.samples, sample_rate=sample_rate, config=config
                ),
            )
        )
    group_lookup = {group.group_id: group for group in spec.groups}
    group_summaries = build_group_summaries(rows, spec.groups)
    all_mel = np.concatenate([bundle.log_mel_power_db.ravel() for _, bundle in rows])
    finite_mel = all_mel[np.isfinite(all_mel)]
    mel_limits = (
        float(np.percentile(finite_mel, 5.0)),
        float(np.percentile(finite_mel, 99.5)),
    )
    generated = ["frame_features.csv", "clip_summary.csv", "group_summary.csv"]
    _write_frame_csv(output_dir / "frame_features.csv", rows)
    _write_clip_csv(output_dir / "clip_summary.csv", rows, group_lookup)
    _write_group_csv(output_dir / "group_summary.csv", group_summaries)
    for clip, bundle in rows:
        name = f"{clip.clip_id}_acoustic_features.png"
        _plot_clip(
            output_dir / name,
            clip,
            bundle,
            group_lookup[clip.group_id].display_color_hex,
            mel_limits,
        )
        generated.append(name)
    _plot_overview(output_dir / "comparison_overview.png", rows, group_lookup)
    generated.append("comparison_overview.png")

    source_hashes: dict[Path, str] = {}
    clips_payload = []
    mappings = []
    for clip, decoded, (_, bundle) in zip(spec.clips, decoded_rows, rows):
        if decoded.path not in source_hashes:
            source_hashes[decoded.path] = sha256_file(decoded.path)
        group = group_lookup[clip.group_id]
        clips_payload.append(
            {
                "clip_id": clip.clip_id,
                "clip_label": clip.clip_label,
                "group_id": clip.group_id,
                "group_label": group.group_label,
                "display_color_hex": group.display_color_hex,
                "label_origin": clip.label_origin,
                "source": {
                    "path": str(decoded.path),
                    "size_bytes": decoded.path.stat().st_size,
                    "sha256": source_hashes[decoded.path],
                    "codec": decoded.source_codec,
                    "sample_rate": decoded.source_sample_rate,
                    "channels": decoded.source_channels,
                    "audio_time_base": decoded.source_time_base,
                    "stream_start_raw_s": decoded.source_stream_start_raw_s,
                },
                "selection": {
                    "start_s": clip.start_s,
                    "end_s": clip.end_s,
                    "metadata": clip.metadata,
                },
                "summary": bundle.summary,
            }
        )
        mappings.append(
            {
                "clip_id": clip.clip_id,
                "group_id": clip.group_id,
                "source_start_s": clip.start_s,
                "source_end_s_requested": clip.end_s,
                "analysis_sample_rate": bundle.sample_rate,
                "analysis_sample_count": len(bundle.samples),
                "decoded_duration_s": len(bundle.samples) / bundle.sample_rate,
                "mapping": "source_time_s = source_start_s + analysis_sample_index / analysis_sample_rate",
                "frame_mapping_table": "frame_features.csv",
            }
        )
    implementation = implementation_provenance()
    payload = {
        "schema_version": ACOUSTIC_SCHEMA_VERSION,
        "analysis_type": "non_biometric_descriptive_acoustic_features",
        "input_manifest": {
            "path": str(spec.manifest_path),
            "sha256": spec.manifest_sha256,
        },
        "boundary": {
            "statement": BOUNDARY_TEXT,
            "aggregate_score_produced": False,
            "aggregate_ranking_produced": False,
            "speaker_embeddings_used": False,
            "voiceprints_used": False,
            "speaker_verification_used": False,
            "same_person_score_produced": False,
            "cross_group_similarity_produced": False,
            "identity_inference_produced": False,
            "group_labels_inferred_from_audio": False,
            "language_labels_inferred_from_audio": False,
            "nationality_inference_produced": False,
            "affiliation_inference_produced": False,
        },
        "method": {
            "config": asdict(config),
            "implementation": implementation,
            "mono_signal": "arithmetic mean of decoded L/R",
            "waveform_plot": "per-clip peak-normalized for display only",
            "log_mel": "Hann-window power spectrum with one shared display scale",
            "f0": "normalized autocorrelation peak; low-periodicity frames are missing",
            "periodicity": "normalized autocorrelation at the selected candidate lag",
            "activity": "adaptive RMS threshold clamped to -55..-30 dBFS",
            "group_summary": "unweighted min, max, median over per-clip scalar summaries",
            "known_confounds": [
                "language and phonetic content",
                "prosody and emotion",
                "microphone and room response",
                "noise suppression and automatic gain control",
                "codec and conferencing mix",
                "overlap and background sound",
            ],
        },
        "metric_definitions": [asdict(definition) for definition in METRICS],
        "clips": clips_payload,
        "group_summaries": group_summaries,
        "sample_mappings": mappings,
    }
    _write_json(output_dir / "acoustic_features.json", payload)
    _write_json(
        output_dir / "sample_mapping.json",
        {
            "schema_version": ACOUSTIC_SCHEMA_VERSION,
            "mappings": mappings,
            "boundary": BOUNDARY_TEXT,
        },
    )
    generated.extend(["acoustic_features.json", "sample_mapping.json"])
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": ACOUSTIC_SCHEMA_VERSION,
            "analysis_type": payload["analysis_type"],
            "input_manifest": payload["input_manifest"],
            "implementation": implementation,
            "artifacts": _hashes(output_dir, generated),
            "boundary": BOUNDARY_TEXT,
        },
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = extract(args.manifest, args.output_dir, args.sample_rate)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"Wrote descriptive acoustic artifacts for {len(payload['clips'])} clips "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
