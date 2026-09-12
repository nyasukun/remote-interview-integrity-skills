from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from acoustics.designated_centered import build_payload, write_outputs
from acoustics.extract_acoustic_features import extract
from acoustics.model import METRICS, scalar_metrics
from acoustics.run_analysis import _preflight_groups, run
from acoustics.signal_features import FeatureConfig, analyze_signal, sha256_file
from acoustics.verify_artifacts import verify_acoustic_dir, verify_designated_dir


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "acoustics-test-matplotlib")
)


def _stereo_tone(
    frequency_hz: float,
    *,
    sample_rate: int,
    duration_s: float,
    left_amplitude: float = 0.20,
    right_amplitude: float = 0.10,
) -> np.ndarray:
    time_s = np.arange(int(round(sample_rate * duration_s))) / sample_rate
    ramp_samples = max(1, int(round(0.01 * sample_rate)))
    envelope = np.ones_like(time_s)
    envelope[:ramp_samples] = np.linspace(0.0, 1.0, ramp_samples)
    envelope[-ramp_samples:] = np.linspace(1.0, 0.0, ramp_samples)
    phase = 2.0 * np.pi * frequency_hz * time_s
    return np.column_stack(
        (
            left_amplitude * envelope * np.sin(phase),
            right_amplitude * envelope * np.sin(phase),
        )
    )


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


class SignalFeatureTests(unittest.TestCase):
    def test_tone_f0_periodicity_stereo_and_spectral_metrics(self) -> None:
        sample_rate = 16_000
        samples = _stereo_tone(180.0, sample_rate=sample_rate, duration_s=0.8)
        bundle = analyze_signal(
            samples,
            sample_rate=sample_rate,
            config=FeatureConfig(sample_rate=sample_rate),
        )
        summary = bundle.summary
        self.assertAlmostEqual(summary["f0_hz"]["distribution"]["median"], 180.0, delta=2.0)
        self.assertGreater(summary["periodicity"]["median"], 0.85)
        self.assertGreater(summary["activity"]["active_frame_fraction"], 0.90)
        self.assertGreater(summary["f0_hz"]["voiced_fraction_of_active"], 0.90)
        self.assertAlmostEqual(summary["stereo"]["balance_db"]["median"], 6.02, delta=0.12)
        self.assertGreater(summary["stereo"]["correlation"]["median"], 0.999)
        self.assertAlmostEqual(summary["stereo"]["side_fraction"]["median"], 0.1, delta=0.01)
        self.assertGreater(
            summary["spectral"]["mean_active_band_fractions"]["low_0_500"],
            0.95,
        )
        self.assertEqual(bundle.log_mel_power_db.shape[1], 64)
        self.assertEqual(set(scalar_metrics(summary)), {metric.metric_id for metric in METRICS})

    def test_mono_stereo_diagnostics_are_explicitly_unavailable(self) -> None:
        sample_rate = 16_000
        mono = _stereo_tone(
            190.0, sample_rate=sample_rate, duration_s=0.5
        )[:, :1]
        bundle = analyze_signal(mono, sample_rate=sample_rate)
        self.assertIs(bundle.summary["stereo"]["available"], False)
        scalars = scalar_metrics(bundle.summary)
        self.assertIsNone(scalars["stereo_balance_median_db"])
        self.assertIsNone(scalars["stereo_correlation_median"])
        self.assertIsNone(scalars["stereo_side_fraction_median"])


class AcousticWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.sample_rate = 16_000
        tones = {
            "anchor.wav": 180.0,
            "set-a-one.wav": 150.0,
            "set-a-two.wav": 210.0,
            "set-b.wav": 255.0,
            "set-c.wav": 180.0,
        }
        for name, frequency in tones.items():
            _write_wav(
                cls.root / name,
                _stereo_tone(
                    frequency,
                    sample_rate=cls.sample_rate,
                    duration_s=0.62,
                ),
                cls.sample_rate,
            )
        cls.manifest_path = cls.root / "input_manifest.json"
        cls.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "groups": [
                        {
                            "id": "focus.1",
                            "label": "Focus label",
                            "color": "#112233",
                        },
                        {
                            "id": "control.alpha",
                            "label": "Alpha label",
                            "color": "#225588",
                        },
                        {
                            "id": "control-beta",
                            "label": "Beta label",
                            "color": "#AA5500",
                        },
                        {
                            "id": "control_gamma",
                            "label": "Gamma label",
                            "color": "#338844",
                        },
                    ],
                    "clips": [
                        {
                            "id": "focus.clip",
                            "group": "focus.1",
                            "source": "anchor.wav",
                            "start_s": 0.01,
                            "end_s": 0.60,
                        },
                        {
                            "id": "alpha-1",
                            "group": "control.alpha",
                            "source": "set-a-one.wav",
                            "start_s": 0.01,
                            "end_s": 0.60,
                        },
                        {
                            "id": "alpha-2",
                            "group": "control.alpha",
                            "source": "set-a-two.wav",
                            "start_s": 0.01,
                            "end_s": 0.60,
                        },
                        {
                            "id": "beta-1",
                            "group": "control-beta",
                            "source": "set-b.wav",
                            "start_s": 0.01,
                            "end_s": 0.60,
                        },
                        {
                            "id": "gamma-1",
                            "group": "control_gamma",
                            "source": "set-c.wav",
                            "start_s": 0.01,
                            "end_s": 0.60,
                        },
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        cls.acoustic_dir = cls.root / "acoustic-output"
        cls.acoustic_payload = extract(
            cls.manifest_path, cls.acoustic_dir, cls.sample_rate
        )
        cls.acoustic_json = cls.acoustic_dir / "acoustic_features.json"
        cls.designated_dir = cls.root / "designated-output"
        cls.designated_payload = build_payload(
            cls.acoustic_json,
            designated_group_id="focus.1",
            comparison_group_ids=(
                "control.alpha",
                "control-beta",
                "control_gamma",
            ),
            include_nearest_median=True,
            expected_input_sha256=sha256_file(cls.acoustic_json),
        )
        write_outputs(cls.designated_payload, cls.designated_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_extractor_artifacts_and_arbitrary_groups(self) -> None:
        payload = self.acoustic_payload
        self.assertEqual(
            [group["group_id"] for group in payload["group_summaries"]],
            ["focus.1", "control.alpha", "control-beta", "control_gamma"],
        )
        alpha = next(
            group
            for group in payload["group_summaries"]
            if group["group_id"] == "control.alpha"
        )
        self.assertEqual(alpha["clip_count"], 2)
        self.assertLess(
            alpha["metrics"]["f0_median_hz"]["min"],
            alpha["metrics"]["f0_median_hz"]["max"],
        )
        self.assertEqual(
            set(alpha["metrics"]), {definition.metric_id for definition in METRICS}
        )
        self.assertIs(payload["boundary"]["aggregate_score_produced"], False)
        self.assertIs(payload["boundary"]["speaker_embeddings_used"], False)
        self.assertEqual(
            payload["group_summaries"][0]["display_color_hex"], "#112233"
        )
        implementation = payload["method"]["implementation"]
        self.assertEqual(implementation["schema_version"], 1)
        self.assertEqual(len(implementation["bundle_sha256"]), 64)
        self.assertIn("signal_features.py", implementation["source_files"])
        self.assertIn("input_manifest.py", implementation["source_files"])
        self.assertIn("python_version", implementation["runtime"])
        self.assertIn("numpy", implementation["runtime"]["distributions"])
        artifact_manifest = json.loads(
            (self.acoustic_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(artifact_manifest["implementation"], implementation)
        required = {
            "acoustic_features.json",
            "frame_features.csv",
            "clip_summary.csv",
            "group_summary.csv",
            "comparison_overview.png",
            "sample_mapping.json",
            "artifact_manifest.json",
        }
        self.assertTrue(required.issubset({path.name for path in self.acoustic_dir.iterdir()}))
        self.assertEqual(verify_acoustic_dir(self.acoustic_dir)["status"], "PASS")

    def test_designated_signed_deltas_ranges_and_optional_nearest(self) -> None:
        payload = self.designated_payload
        self.assertEqual(payload["configuration"]["designated_group_id"], "focus.1")
        self.assertEqual(len(payload["metrics"]), len(METRICS))
        f0 = next(metric for metric in payload["metrics"] if metric["metric_id"] == "f0_median_hz")
        designated = f0["designated"]["value"]
        alpha = f0["comparisons"][0]
        self.assertAlmostEqual(
            alpha["signed_delta_group_median_minus_designated"],
            alpha["median"] - designated,
            places=12,
        )
        self.assertIs(alpha["designated_inside_inclusive_range"], True)
        self.assertEqual(
            f0["nearest_group_median_for_this_metric"], ["control_gamma"]
        )
        self.assertIs(f0["comparisons"][2]["is_nearest_median_for_this_metric"], True)
        self.assertIs(payload["non_composability"]["composite_valid"], False)
        for forbidden in (
            "aggregate_score",
            "overall_rank",
            "speaker_similarity",
            "same_speaker_probability",
        ):
            self.assertFalse(_contains_key(payload, forbidden))
        self.assertEqual(verify_designated_dir(self.designated_dir)["status"], "PASS")

    def test_nearest_median_can_be_omitted(self) -> None:
        payload = build_payload(
            self.acoustic_json,
            designated_group_id="focus.1",
            comparison_group_ids=("control.alpha",),
            include_nearest_median=False,
        )
        for metric in payload["metrics"]:
            self.assertNotIn("nearest_group_median_for_this_metric", metric)
            self.assertNotIn(
                "is_nearest_median_for_this_metric", metric["comparisons"][0]
            )

    def test_selection_constraints_fail_closed(self) -> None:
        cases = (
            ("control.alpha", ("control-beta",)),
            ("focus.1", ()),
            (
                "focus.1",
                ("control.alpha", "control-beta", "control_gamma", "missing"),
            ),
            ("focus.1", ("control.alpha", "control.alpha")),
            ("focus.1", ("focus.1",)),
        )
        for designated, comparisons in cases:
            with self.subTest(designated=designated, comparisons=comparisons):
                with self.assertRaises(ValueError):
                    build_payload(
                        self.acoustic_json,
                        designated_group_id=designated,
                        comparison_group_ids=comparisons,
                    )

    def test_source_boundary_and_output_overwrite_fail_closed(self) -> None:
        tampered = self.root / "tampered-source.json"
        source = json.loads(self.acoustic_json.read_text(encoding="utf-8"))
        source["boundary"]["speaker_embeddings_used"] = True
        tampered.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "speaker_embeddings_used"):
            build_payload(
                tampered,
                designated_group_id="focus.1",
                comparison_group_ids=("control.alpha",),
            )
        with self.assertRaises(FileExistsError):
            write_outputs(self.designated_payload, self.designated_dir)

    def test_verifier_detects_tampered_artifact(self) -> None:
        copied = self.root / "tampered-designated"
        shutil.copytree(self.designated_dir, copied)
        with (copied / "designated_centered_comparison.csv").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("tampered\n")
        result = verify_designated_dir(copied)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("artifact_hash" in error for error in result["errors"]))

    def test_verifier_rejects_forged_implementation_provenance(self) -> None:
        copied = self.root / "forged-acoustic"
        shutil.copytree(self.acoustic_dir, copied)
        payload_path = copied / "acoustic_features.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["method"]["implementation"]["bundle_sha256"] = "0" * 64
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        manifest_path = copied / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["implementation"] = payload["method"]["implementation"]
        manifest["artifacts"]["acoustic_features.json"] = sha256_file(payload_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = verify_acoustic_dir(copied)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("implementation_or_runtime_mismatch", result["errors"])

    def test_orchestrator_preflight_accepts_generic_ids(self) -> None:
        _preflight_groups(
            self.manifest_path,
            "focus.1",
            ("control.alpha", "control-beta", "control_gamma"),
        )
        with self.assertRaises(ValueError):
            _preflight_groups(
                self.manifest_path,
                "control.alpha",
                ("control-beta",),
            )

    def test_orchestrator_runs_stages_and_records_input_and_config_hashes(self) -> None:
        output = self.root / "workflow-output"
        report = run(
            self.manifest_path,
            output,
            designated_group_id="focus.1",
            comparison_group_ids=("control.alpha",),
            sample_rate=self.sample_rate,
            include_nearest_median=False,
        )
        self.assertEqual(report["verification"]["acoustic"]["status"], "PASS")
        self.assertEqual(
            report["verification"]["designated_centered"]["status"], "PASS"
        )
        self.assertEqual(
            report["input_manifest"]["sha256"], sha256_file(self.manifest_path)
        )
        self.assertEqual(
            report["configuration"]["sha256"],
            sha256_file(output / "run_configuration.json"),
        )
        self.assertTrue((output / "run_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
