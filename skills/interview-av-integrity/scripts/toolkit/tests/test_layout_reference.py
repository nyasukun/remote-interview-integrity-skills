from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[3]
TOOLKIT_ROOT = SKILL_ROOT / "scripts" / "toolkit"
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from video_integrity_analyzer.layout_reference import (  # noqa: E402
    APPROVED_REFERENCE_SHA256,
    REQUIRED_QUALITY_CHECKS,
    LayoutReferenceError,
    load_approved_layout_reference,
    sha256_file,
    verify_approved_preview,
)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREVIEW = _load_script(
    "av_layout_preview_test_module",
    SKILL_ROOT / "scripts" / "render_layout_preview.py",
)
RENDERER = _load_script(
    "av_layout_renderer_test_module",
    SKILL_ROOT
    / "scripts"
    / "toolkit"
    / "scripts"
    / "render_closure_evidence_video.py",
)


def _input_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "title": "Synthetic preview fixture",
                "events": [
                    {
                        "event_id": "layout-fixture",
                        "source_release_s": 2.0,
                        "classification": "closure_absent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class LayoutReferenceTests(unittest.TestCase):
    def test_approved_asset_manifest_and_image_are_exact(self) -> None:
        reference = load_approved_layout_reference()
        self.assertEqual(reference["sha256"], APPROVED_REFERENCE_SHA256)
        self.assertEqual((reference["width"], reference["height"]), (1672, 941))
        self.assertEqual(reference["color_mode"], "RGB")
        self.assertEqual(reference["provenance"], "user_provided_synthetic_layout_reference")
        self.assertFalse(reference["case_evidence"])

    def test_production_path_preview_and_complete_approval_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "events.json"
            preview = root / "preview.png"
            sheet = root / "review.png"
            provenance = root / "provenance.json"
            _input_manifest(manifest)
            report = PREVIEW.render_layout_preview(
                manifest,
                preview,
                review_sheet=sheet,
                report=provenance,
            )
            self.assertEqual(report["review_status"], "PENDING_USER_APPROVAL")
            self.assertFalse(report["case_media_used"])
            self.assertEqual(
                report["quality_contract"]["required_human_checks"],
                list(REQUIRED_QUALITY_CHECKS),
            )
            self.assertEqual(
                (report["implementation_preview"]["width"], report["implementation_preview"]["height"]),
                (1920, 1080),
            )
            self.assertEqual(
                (report["side_by_side_review_sheet"]["width"], report["side_by_side_review_sheet"]["height"]),
                (1920, 660),
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["layout_review"] = {
                "status": "APPROVED",
                "approval_basis": "user explicitly approved the displayed synthetic production preview",
                "preview": {"path": preview.name, "sha256": sha256_file(preview)},
                "preview_provenance": {
                    "path": provenance.name,
                    "sha256": sha256_file(provenance),
                },
                "review_sheet": {"path": sheet.name, "sha256": sha256_file(sheet)},
                "quality_checks": {
                    name: True for name in REQUIRED_QUALITY_CHECKS
                },
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            approved = verify_approved_preview(manifest)
            self.assertEqual(approved["status"], "APPROVED")
            self.assertEqual(approved["preview"]["sha256"], sha256_file(preview))

    def test_incomplete_or_tampered_approval_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "events.json"
            preview = root / "preview.png"
            sheet = root / "review.png"
            provenance = root / "provenance.json"
            _input_manifest(manifest)
            PREVIEW.render_layout_preview(
                manifest,
                preview,
                review_sheet=sheet,
                report=provenance,
            )
            with self.assertRaisesRegex(LayoutReferenceError, "layout_review is required"):
                verify_approved_preview(manifest)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            checks = {name: True for name in REQUIRED_QUALITY_CHECKS}
            checks[REQUIRED_QUALITY_CHECKS[0]] = False
            payload["layout_review"] = {
                "status": "APPROVED",
                "approval_basis": "synthetic test approval",
                "preview": {"path": str(preview), "sha256": sha256_file(preview)},
                "preview_provenance": {
                    "path": str(provenance),
                    "sha256": sha256_file(provenance),
                },
                "review_sheet": {"path": str(sheet), "sha256": sha256_file(sheet)},
                "quality_checks": checks,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LayoutReferenceError, "incomplete"):
                verify_approved_preview(manifest)

            payload["layout_review"]["quality_checks"] = {
                name: True for name in REQUIRED_QUALITY_CHECKS
            }
            payload["layout_review"]["preview"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LayoutReferenceError, "preview SHA-256 mismatch"):
                verify_approved_preview(manifest)

    def test_full_render_cli_rejects_unapproved_layout_before_media_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "events.json"
            _input_manifest(manifest)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = RENDERER.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--video",
                        str(root / "missing-case-video.mp4"),
                        "--output",
                        str(root / "should-not-render.mp4"),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertIn("layout_review is required", stderr.getvalue())
            self.assertNotIn("source video not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
