from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[3] / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("voice_check_environment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


class EnvironmentCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = Mock(side_effect=lambda name, mode: SimpleNamespace(name=name))
        self.dependencies = {
            import_name: SimpleNamespace(__version__="test-version")
            for import_name in environment.REQUIRED_MODULES.values()
        }
        self.dependencies["av"].codec = SimpleNamespace(Codec=self.codec)
        imports = patch.object(
            environment.importlib, "import_module", side_effect=self.dependencies.__getitem__
        )
        imports.start()
        self.addCleanup(imports.stop)

    def test_default_check_preserves_report_schema_and_encoder_checks(self) -> None:
        report = environment.check()
        self.assertEqual(set(report), {
            "schema_version", "status", "python", "modules", "encoders",
            "network_required", "failures",
        })
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["encoders"], {
            "libx264": {"status": "PASS", "canonical_name": "libx264"},
            "aac": {"status": "PASS", "canonical_name": "aac"},
        })
        self.assertEqual(set(report["modules"]), set(environment.REQUIRED_MODULES))

    def test_encoder_unavailability_fails_default_but_permits_analysis_only(self) -> None:
        self.codec.side_effect = RuntimeError("encoder unavailable")
        full = environment.check()
        self.assertEqual(full["status"], "FAIL")
        self.assertEqual(len(full["failures"]), 2)
        self.assertTrue(all(item["status"] == "FAIL" for item in full["encoders"].values()))

        self.codec.reset_mock()
        analysis = environment.check(analysis_only=True)
        self.assertEqual(analysis["status"], "PASS")
        self.assertEqual(analysis["encoders"], {})
        self.assertEqual(analysis["failures"], [])
        self.codec.assert_not_called()

    def test_missing_analysis_dependency_fails_both_modes(self) -> None:
        for package, import_name in environment.REQUIRED_MODULES.items():
            def import_dependency(name: str):
                if name == import_name:
                    raise ModuleNotFoundError(name)
                return self.dependencies[name]

            for analysis_only in (False, True):
                with self.subTest(package=package, analysis_only=analysis_only):
                    with patch.object(environment.importlib, "import_module", side_effect=import_dependency):
                        report = environment.check(analysis_only=analysis_only)
                    self.assertEqual(report["status"], "FAIL")
                    self.assertEqual(report["modules"][package]["status"], "FAIL")
                    self.assertTrue(any(f"cannot import {package}:" in reason for reason in report["failures"]))

    def test_unsupported_python_fails_both_modes(self) -> None:
        with patch.object(environment.sys, "version_info", (3, 10)):
            for analysis_only in (False, True):
                with self.subTest(analysis_only=analysis_only):
                    report = environment.check(analysis_only=analysis_only)
                    self.assertEqual(report["status"], "FAIL")
                    self.assertIn("Python 3.11 or newer is required", report["failures"])

    def test_cli_analysis_only_flag_controls_exit_status(self) -> None:
        self.codec.side_effect = RuntimeError("encoder unavailable")
        for argv, expected_status, expected_exit in (
            ([], "FAIL", 1),
            (["--analysis-only"], "PASS", 0),
        ):
            with self.subTest(argv=argv):
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = environment.main(argv)
                self.assertEqual(exit_code, expected_exit)
                self.assertEqual(json.loads(output.getvalue())["status"], expected_status)


if __name__ == "__main__":
    unittest.main()
