"""Validate the fixed, synthetic A/V evidence-video layout reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


SKILL_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_RELATIVE_PATH = Path(
    "assets/layout-references/av-integrity-closure-review-approved.png"
)
MANIFEST_RELATIVE_PATH = Path("assets/layout-references/manifest.json")
APPROVED_REFERENCE_ID = "av-integrity-closure-review-approved-v1"
APPROVED_REFERENCE_SHA256 = (
    "a847a47025555f72ae83046de1169f0022105f4ef10c9028a52757200020b84c"
)
APPROVED_REFERENCE_BYTES = 1_778_626
APPROVED_REFERENCE_SIZE = (1672, 941)
APPROVED_REFERENCE_MODE = "RGB"
PRODUCTION_SIZE = (1920, 1080)
REVIEW_SHEET_SIZE = (1920, 660)
REQUIRED_QUALITY_CHECKS = (
    "reference_viewed_first",
    "visual_hierarchy_and_density_match",
    "case_finding_speed_and_legend_readable",
    "source_panel_remains_dominant",
    "mouth_roi_and_same_frame_inset_traceable",
    "mouth_crop_contains_lips_and_jaw",
    "closure_window_and_burst_emphasis_readable",
    "release_marker_and_playhead_readable",
    "japanese_text_readable_at_1920x1080",
    "limitations_and_uncertainty_readable",
    "production_renderer_preview",
    "side_by_side_review_completed",
)


class LayoutReferenceError(ValueError):
    """Raised when the approved layout reference or its provenance changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_manifest_basis_sha256(manifest: Path) -> str:
    """Hash the input event manifest without the post-preview approval block."""

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayoutReferenceError(f"cannot read input event manifest: {error}") from error
    if not isinstance(payload, dict):
        raise LayoutReferenceError("input event manifest must be an object")
    basis = dict(payload)
    basis.pop("layout_review", None)
    return _canonical_sha256(basis)


def renderer_source_records(skill_root: Path | None = None) -> list[dict[str, str]]:
    root = (skill_root or SKILL_ROOT).expanduser().resolve()
    sources = (
        root / "scripts/render_layout_preview.py",
        root / "scripts/toolkit/scripts/render_closure_evidence_video.py",
        Path(__file__).resolve(),
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise LayoutReferenceError(f"layout renderer source is missing: {missing}")
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in sources
    ]


def _resolved_file(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LayoutReferenceError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise LayoutReferenceError(f"{label} does not exist: {path}")
    return path


def _image_properties(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            return {
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            }
    except OSError as error:
        raise LayoutReferenceError(f"cannot decode layout image {path}: {error}") from error


def _load_manifest(skill_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = skill_root / MANIFEST_RELATIVE_PATH
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayoutReferenceError(
            f"cannot read approved layout manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise LayoutReferenceError("approved layout manifest root must be an object")
    return manifest_path, payload


def load_approved_layout_reference(
    skill_root: Path | None = None,
) -> dict[str, Any]:
    """Return verified, relative provenance for the approved layout asset."""

    root = (skill_root or SKILL_ROOT).expanduser().resolve()
    manifest_path, payload = _load_manifest(root)
    if payload.get("schema_version") != 1:
        raise LayoutReferenceError("approved layout manifest schema_version must be 1")
    if payload.get("reference_id") != APPROVED_REFERENCE_ID:
        raise LayoutReferenceError("approved layout reference_id changed")
    if payload.get("status") != "approved_starting_layout":
        raise LayoutReferenceError("approved layout status changed")
    if payload.get("provenance") != "user_provided_synthetic_layout_reference":
        raise LayoutReferenceError("approved layout provenance changed")
    if payload.get("case_evidence") is not False:
        raise LayoutReferenceError("approved layout must not be represented as case evidence")
    if payload.get("authoritative_scope") != "layout_structure_and_visual_quality_only":
        raise LayoutReferenceError("approved layout authoritative scope changed")
    asset = payload.get("asset")
    if not isinstance(asset, dict):
        raise LayoutReferenceError("approved layout manifest.asset must be an object")
    expected_asset = {
        "path": REFERENCE_RELATIVE_PATH.as_posix(),
        "sha256": APPROVED_REFERENCE_SHA256,
        "byte_size": APPROVED_REFERENCE_BYTES,
        "width": APPROVED_REFERENCE_SIZE[0],
        "height": APPROVED_REFERENCE_SIZE[1],
        "color_mode": APPROVED_REFERENCE_MODE,
    }
    for key, expected in expected_asset.items():
        if asset.get(key) != expected:
            raise LayoutReferenceError(
                f"approved layout manifest asset.{key} changed: "
                f"{asset.get(key)!r} != {expected!r}"
            )
    quality_gate = payload.get("quality_gate")
    required_gate = {
        "new_imagegen_concept_allowed": False,
        "production_renderer_preview_required": True,
        "side_by_side_review_required": True,
        "explicit_user_approval_required_before_full_render": True,
        "case_media_allowed_in_preview": False,
    }
    if not isinstance(quality_gate, dict):
        raise LayoutReferenceError("approved layout quality_gate must be an object")
    for key, expected in required_gate.items():
        if quality_gate.get(key) is not expected:
            raise LayoutReferenceError(
                f"approved layout quality_gate.{key} changed"
            )

    asset_path = root / REFERENCE_RELATIVE_PATH
    if not asset_path.is_file():
        raise LayoutReferenceError(f"approved layout asset is missing: {asset_path}")
    if asset_path.stat().st_size != APPROVED_REFERENCE_BYTES:
        raise LayoutReferenceError("approved layout asset byte size changed")
    if sha256_file(asset_path) != APPROVED_REFERENCE_SHA256:
        raise LayoutReferenceError("approved layout asset SHA-256 changed")
    try:
        with Image.open(asset_path) as image:
            image.load()
            if image.size != APPROVED_REFERENCE_SIZE:
                raise LayoutReferenceError("approved layout asset dimensions changed")
            if image.mode != APPROVED_REFERENCE_MODE:
                raise LayoutReferenceError("approved layout asset color mode changed")
    except OSError as error:
        raise LayoutReferenceError(
            f"cannot decode approved layout asset {asset_path}: {error}"
        ) from error

    return {
        "reference_id": APPROVED_REFERENCE_ID,
        "role": "authoritative_starting_layout",
        "provenance": "user_provided_synthetic_layout_reference",
        "case_evidence": False,
        "authoritative_scope": "layout_structure_and_visual_quality_only",
        "relative_path": REFERENCE_RELATIVE_PATH.as_posix(),
        "asset_manifest_relative_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "sha256": APPROVED_REFERENCE_SHA256,
        "asset_manifest_sha256": sha256_file(manifest_path),
        "byte_size": APPROVED_REFERENCE_BYTES,
        "width": APPROVED_REFERENCE_SIZE[0],
        "height": APPROVED_REFERENCE_SIZE[1],
        "color_mode": APPROVED_REFERENCE_MODE,
        "quality_gate": required_gate,
    }


def validate_effective_layout_reference(
    value: Mapping[str, Any] | None,
    skill_root: Path | None = None,
) -> dict[str, Any]:
    """Require effective-manifest provenance to equal the installed asset."""

    if not isinstance(value, Mapping):
        raise LayoutReferenceError(
            "render.layout_reference must contain approved layout provenance"
        )
    expected = load_approved_layout_reference(skill_root)
    if dict(value) != expected:
        raise LayoutReferenceError(
            "render.layout_reference differs from the approved installed asset"
        )
    return expected


def verify_approved_preview(
    input_manifest: Path,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a production preview has an explicit complete approval."""

    manifest_path = input_manifest.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayoutReferenceError(f"cannot read input event manifest: {error}") from error
    if not isinstance(payload, dict):
        raise LayoutReferenceError("input event manifest must be an object")
    review = payload.get("layout_review")
    if not isinstance(review, dict):
        raise LayoutReferenceError(
            "layout_review is required before production rendering; generate and inspect the production preview first"
        )
    if review.get("status") != "APPROVED":
        raise LayoutReferenceError(
            "layout_review.status must be APPROVED after explicit user approval"
        )
    approval_basis = review.get("approval_basis")
    if not isinstance(approval_basis, str) or not approval_basis.strip():
        raise LayoutReferenceError(
            "layout_review.approval_basis must record the explicit user approval"
        )
    checks = review.get("quality_checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_QUALITY_CHECKS):
        raise LayoutReferenceError(
            "layout_review.quality_checks must contain the exact required checklist"
        )
    failed = [name for name in REQUIRED_QUALITY_CHECKS if checks.get(name) is not True]
    if failed:
        raise LayoutReferenceError(
            "layout review is incomplete: " + ", ".join(failed)
        )

    entries: dict[str, Path] = {}
    for key, label in (
        ("preview", "layout implementation preview"),
        ("preview_provenance", "layout preview provenance"),
        ("review_sheet", "layout side-by-side review sheet"),
    ):
        entry = review.get(key)
        if not isinstance(entry, dict):
            raise LayoutReferenceError(f"layout_review.{key} must be an object")
        path = _resolved_file(manifest_path.parent, entry.get("path"), label)
        if entry.get("sha256") != sha256_file(path):
            raise LayoutReferenceError(f"layout_review.{key} SHA-256 mismatch")
        entries[key] = path

    preview_properties = _image_properties(entries["preview"])
    if preview_properties != {
        "format": "PNG",
        "mode": "RGB",
        "width": PRODUCTION_SIZE[0],
        "height": PRODUCTION_SIZE[1],
    }:
        raise LayoutReferenceError("approved implementation preview must be 1920x1080 RGB PNG")
    sheet_properties = _image_properties(entries["review_sheet"])
    if sheet_properties != {
        "format": "PNG",
        "mode": "RGB",
        "width": REVIEW_SHEET_SIZE[0],
        "height": REVIEW_SHEET_SIZE[1],
    }:
        raise LayoutReferenceError("approved review sheet properties changed")

    try:
        provenance = json.loads(
            entries["preview_provenance"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise LayoutReferenceError(f"cannot read layout preview provenance: {error}") from error
    if not isinstance(provenance, dict):
        raise LayoutReferenceError("layout preview provenance must be an object")
    if provenance.get("schema_version") != 1:
        raise LayoutReferenceError("layout preview provenance schema_version must be 1")
    if provenance.get("kind") != "av_integrity_production_renderer_layout_preview":
        raise LayoutReferenceError("layout preview provenance kind changed")
    if provenance.get("review_status") != "PENDING_USER_APPROVAL":
        raise LayoutReferenceError(
            "preview provenance must remain the immutable pre-approval record"
        )
    expected_basis = render_manifest_basis_sha256(manifest_path)
    if provenance.get("render_manifest_basis_sha256") != expected_basis:
        raise LayoutReferenceError(
            "layout preview was made from a different input-manifest basis"
        )
    preview_record = provenance.get("implementation_preview")
    if not isinstance(preview_record, dict):
        raise LayoutReferenceError("preview provenance lacks implementation_preview")
    if Path(str(preview_record.get("path", ""))).resolve() != entries["preview"]:
        raise LayoutReferenceError("preview provenance image path mismatch")
    if preview_record.get("sha256") != sha256_file(entries["preview"]):
        raise LayoutReferenceError("preview provenance image SHA-256 mismatch")
    if (preview_record.get("width"), preview_record.get("height")) != PRODUCTION_SIZE:
        raise LayoutReferenceError("preview provenance dimensions are not 1920x1080")
    sheet_record = provenance.get("side_by_side_review_sheet")
    if not isinstance(sheet_record, dict):
        raise LayoutReferenceError("preview provenance lacks side_by_side_review_sheet")
    if Path(str(sheet_record.get("path", ""))).resolve() != entries["review_sheet"]:
        raise LayoutReferenceError("review-sheet provenance path mismatch")
    if sheet_record.get("sha256") != sha256_file(entries["review_sheet"]):
        raise LayoutReferenceError("review-sheet provenance SHA-256 mismatch")
    if (sheet_record.get("width"), sheet_record.get("height")) != REVIEW_SHEET_SIZE:
        raise LayoutReferenceError("review-sheet provenance dimensions changed")

    current_reference = dict(reference or load_approved_layout_reference())
    if provenance.get("layout_reference") != current_reference:
        raise LayoutReferenceError("preview used a different approved layout reference")
    current_sources = renderer_source_records()
    if provenance.get("renderer_sources") != current_sources:
        raise LayoutReferenceError(
            "production renderer sources changed after the approved preview"
        )
    contract = provenance.get("quality_contract")
    if not isinstance(contract, dict):
        raise LayoutReferenceError("preview provenance lacks quality_contract")
    if contract.get("required_human_checks") != list(REQUIRED_QUALITY_CHECKS):
        raise LayoutReferenceError("preview provenance quality checklist changed")
    if provenance.get("case_media_used") is not False:
        raise LayoutReferenceError("preview provenance must state that no case media was used")
    if provenance.get("new_imagegen_concept_used") is not False:
        raise LayoutReferenceError("preview provenance must reject a new ImageGen concept")

    return {
        "status": "APPROVED",
        "approval_basis": approval_basis.strip(),
        "quality_checks": checks,
        "preview": {
            "path": str(entries["preview"]),
            "sha256": sha256_file(entries["preview"]),
            **preview_properties,
        },
        "preview_provenance": {
            "path": str(entries["preview_provenance"]),
            "sha256": sha256_file(entries["preview_provenance"]),
        },
        "review_sheet": {
            "path": str(entries["review_sheet"]),
            "sha256": sha256_file(entries["review_sheet"]),
            **sheet_properties,
        },
        "render_manifest_basis_sha256": expected_basis,
        "renderer_sources": current_sources,
    }


def validate_effective_layout_review(
    value: Mapping[str, Any] | None,
    input_manifest: Path,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-verify approval artifacts and require the effective record to match."""

    if not isinstance(value, Mapping):
        raise LayoutReferenceError(
            "render.layout_review must contain approved preview provenance"
        )
    expected = verify_approved_preview(input_manifest, reference)
    if dict(value) != expected:
        raise LayoutReferenceError(
            "render.layout_review differs from the currently verified approval artifacts"
        )
    return expected
