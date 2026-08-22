"""Integrity and approval gates for the canonical voice-comparison layout reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SKILL_DIR = Path(__file__).resolve().parents[3]
REFERENCE_DIR = SKILL_DIR / "assets" / "layout-references"
REFERENCE_MANIFEST = REFERENCE_DIR / "layout-reference-manifest.json"
EXPECTED_ASSET_NAME = "voice-signal-designated-comparison-approved.png"
EXPECTED_ASSET_SHA256 = "6d85aebbdefa38d886f8f78326e4ebb94b1aa66af797a9a0a4ac03c45311039a"
EXPECTED_ASSET_BYTES = 1_830_370
EXPECTED_ASSET_SIZE = (1672, 941)
EXPECTED_ASSET_MODE = "RGB"
EXPECTED_ASSET_FORMAT = "PNG"
PRODUCTION_SIZE = (1920, 1080)
REVIEW_SHEET_SIZE = (1920, 720)

REQUIRED_QUALITY_CHECKS = (
    "reference_viewed_first",
    "visual_hierarchy_matches",
    "designated_anchor_fixed_left",
    "comparison_panel_on_right",
    "waveform_and_logmel_readable",
    "shared_signed_delta_axes",
    "range_and_median_visible",
    "no_ranking_or_identity_claim",
    "dual_mono_treated_as_non_identifying",
    "limitation_strip_readable",
    "production_renderer_preview",
    "side_by_side_review_sheet_inspected",
)


class LayoutReferenceError(ValueError):
    pass


def sha256(path: Path) -> str:
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


def render_manifest_basis_sha256(manifest: Path) -> str:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LayoutReferenceError("render manifest must be an object")
    payload = dict(payload)
    payload.pop("layout_review", None)
    return _canonical_sha256(payload)


def verify_layout_reference() -> dict[str, Any]:
    if not REFERENCE_MANIFEST.is_file():
        raise LayoutReferenceError(f"layout reference manifest is missing: {REFERENCE_MANIFEST}")
    payload = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise LayoutReferenceError("layout reference manifest must use schema_version 1")
    if payload.get("path") != EXPECTED_ASSET_NAME:
        raise LayoutReferenceError("layout reference asset path differs from the approved allowlist")
    if payload.get("sha256") != EXPECTED_ASSET_SHA256:
        raise LayoutReferenceError("layout reference manifest SHA-256 differs from the approved allowlist")
    if payload.get("size_bytes") != EXPECTED_ASSET_BYTES:
        raise LayoutReferenceError("layout reference manifest byte size differs from the approved allowlist")
    if payload.get("role") != "user-provided synthetic layout reference; not evidence":
        raise LayoutReferenceError("layout reference role must identify the asset as synthetic and not evidence")

    asset = (REFERENCE_DIR / EXPECTED_ASSET_NAME).resolve()
    if not asset.is_file() or asset.parent != REFERENCE_DIR.resolve():
        raise LayoutReferenceError("approved layout reference asset is missing or escaped its directory")
    actual_hash = sha256(asset)
    if actual_hash != EXPECTED_ASSET_SHA256:
        raise LayoutReferenceError("approved layout reference asset SHA-256 mismatch")
    if asset.stat().st_size != EXPECTED_ASSET_BYTES:
        raise LayoutReferenceError("approved layout reference asset byte-size mismatch")
    with Image.open(asset) as image:
        image.load()
        actual = {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "has_alpha": "A" in image.getbands(),
        }
    expected = {
        "format": EXPECTED_ASSET_FORMAT,
        "mode": EXPECTED_ASSET_MODE,
        "width": EXPECTED_ASSET_SIZE[0],
        "height": EXPECTED_ASSET_SIZE[1],
        "has_alpha": False,
    }
    if actual != expected or payload.get("image") != expected:
        raise LayoutReferenceError("approved layout reference image properties mismatch")
    return {
        "asset_id": payload.get("asset_id"),
        "role": payload["role"],
        "asset": {
            "path": str(asset),
            "sha256": actual_hash,
            "size_bytes": asset.stat().st_size,
            **actual,
        },
        "manifest": {
            "path": str(REFERENCE_MANIFEST.resolve()),
            "sha256": sha256(REFERENCE_MANIFEST),
        },
        "production_canvas": {"width": PRODUCTION_SIZE[0], "height": PRODUCTION_SIZE[1]},
        "authority": payload.get("authority"),
        "production_precedence": payload.get("production_precedence"),
    }


def preview_provenance_path(preview: Path) -> Path:
    return preview.with_name(preview.name + ".provenance.json")


def review_sheet_path(preview: Path) -> Path:
    return preview.with_name(preview.stem + ".review-sheet.png")


def _review_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def write_review_sheet(preview: Path, reference: dict[str, Any]) -> Path:
    preview = preview.expanduser().resolve()
    reference_asset = Path(reference["asset"]["path"])
    with Image.open(reference_asset) as source:
        left = source.convert("RGB").resize((930, 523), Image.Resampling.LANCZOS)
    with Image.open(preview) as source:
        right = source.convert("RGB").resize((930, 523), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", REVIEW_SHEET_SIZE, (5, 12, 18))
    draw = ImageDraw.Draw(sheet)
    title_font = _review_font(28, True)
    label_font = _review_font(18, True)
    note_font = _review_font(15)
    draw.text(
        (960, 14),
        "LAYOUT REVIEW — canonical composition reference vs production preview",
        anchor="ma",
        font=title_font,
        fill=(235, 241, 246),
    )
    draw.text(
        (20, 53),
        "APPROVED SYNTHETIC REFERENCE — composition / hierarchy / quality only",
        font=label_font,
        fill=(255, 145, 32),
    )
    draw.text(
        (970, 53),
        "PRODUCTION RENDERER PREVIEW — synthetic waveform / Log-Mel placeholders",
        font=label_font,
        fill=(46, 160, 255),
    )
    sheet.paste(left, (20, 80))
    sheet.paste(right, (970, 80))
    draw.rectangle((19, 79, 950, 603), outline=(255, 145, 32), width=2)
    draw.rectangle((969, 79, 1900, 603), outline=(46, 160, 255), width=2)
    draw.line((20, 620, 1900, 620), fill=(70, 88, 100), width=2)
    draw.text(
        (20, 634),
        "Reference controls visual hierarchy, density, and technical polish. Renderer rules control the actual labels, values, axes, and playback.",
        font=note_font,
        fill=(220, 227, 233),
    )
    draw.text(
        (20, 666),
        "Confirm: designated anchor fixed left; one comparison right; waveform + Log-Mel; shared signed-delta axes; range + median; no ranking; limitation readable.",
        font=note_font,
        fill=(160, 175, 186),
    )
    output = review_sheet_path(preview)
    sheet.save(output)
    return output


def write_preview_provenance(
    preview: Path, render_manifest: Path, reference: dict[str, Any]
) -> Path:
    preview = preview.expanduser().resolve()
    render_manifest = render_manifest.expanduser().resolve()
    with Image.open(preview) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != PRODUCTION_SIZE:
            raise LayoutReferenceError("production preview must be a 1920x1080 RGB PNG")
    review_sheet = write_review_sheet(preview, reference)
    source_files = (
        SKILL_DIR / "scripts" / "render_layout_preview.py",
        SKILL_DIR / "scripts" / "toolkit" / "rendering" / "render.py",
        SKILL_DIR / "scripts" / "toolkit" / "rendering" / "common.py",
        Path(__file__).resolve(),
    )
    payload = {
        "schema_version": 1,
        "kind": "voice_signal_production_renderer_layout_preview",
        "review_status": "PENDING_HUMAN_REVIEW",
        "render_manifest_basis_sha256": render_manifest_basis_sha256(render_manifest),
        "reference": reference,
        "preview": {
            "path": str(preview),
            "sha256": sha256(preview),
            "format": "PNG",
            "mode": "RGB",
            "width": PRODUCTION_SIZE[0],
            "height": PRODUCTION_SIZE[1],
        },
        "review_sheet": {
            "path": str(review_sheet),
            "sha256": sha256(review_sheet),
            "format": "PNG",
            "mode": "RGB",
            "width": REVIEW_SHEET_SIZE[0],
            "height": REVIEW_SHEET_SIZE[1],
        },
        "renderer_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in source_files
        ],
        "quality_contract": {
            "structure_source": "production _clip_frame path with synthetic waveform/Log-Mel placeholders",
            "reference_is_composition_only": True,
            "reference_text_people_values_and_density_curves_are_non_authoritative": True,
            "current_renderer_rules_take_precedence": True,
            "required_human_checks": list(REQUIRED_QUALITY_CHECKS),
        },
    }
    output = preview_provenance_path(preview)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def verify_approved_preview(
    render_manifest: Path, reference: dict[str, Any]
) -> dict[str, Any]:
    render_manifest = render_manifest.expanduser().resolve()
    payload = json.loads(render_manifest.read_text(encoding="utf-8"))
    review = payload.get("layout_review") if isinstance(payload, dict) else None
    if not isinstance(review, dict):
        raise LayoutReferenceError(
            "layout_review is required before production rendering; generate and inspect the production preview first"
        )
    if review.get("status") != "APPROVED":
        raise LayoutReferenceError("layout_review.status must be APPROVED after explicit user approval")
    if not isinstance(review.get("approval_basis"), str) or not review["approval_basis"].strip():
        raise LayoutReferenceError("layout_review.approval_basis must record the explicit user approval")
    checks = review.get("quality_checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_QUALITY_CHECKS):
        raise LayoutReferenceError("layout_review.quality_checks must contain the exact required checklist")
    failed = [name for name in REQUIRED_QUALITY_CHECKS if checks.get(name) is not True]
    if failed:
        raise LayoutReferenceError("layout review is incomplete: " + ", ".join(failed))

    preview_entry = review.get("preview")
    provenance_entry = review.get("preview_provenance")
    if not isinstance(preview_entry, dict) or not isinstance(provenance_entry, dict):
        raise LayoutReferenceError("layout_review must reference preview and preview provenance files")
    preview = _resolved_file(render_manifest.parent, preview_entry.get("path"), "layout preview")
    provenance = _resolved_file(
        render_manifest.parent,
        provenance_entry.get("path"),
        "layout preview provenance",
    )
    sheet_entry = review.get("review_sheet")
    if not isinstance(sheet_entry, dict):
        raise LayoutReferenceError("layout_review must reference the side-by-side review sheet")
    review_sheet = _resolved_file(
        render_manifest.parent, sheet_entry.get("path"), "layout review sheet"
    )
    if preview_entry.get("sha256") != sha256(preview):
        raise LayoutReferenceError("layout preview SHA-256 mismatch")
    if provenance_entry.get("sha256") != sha256(provenance):
        raise LayoutReferenceError("layout preview provenance SHA-256 mismatch")
    if sheet_entry.get("sha256") != sha256(review_sheet):
        raise LayoutReferenceError("layout review sheet SHA-256 mismatch")
    preview_payload = json.loads(provenance.read_text(encoding="utf-8"))
    if preview_payload.get("schema_version") != 1 or preview_payload.get("kind") != "voice_signal_production_renderer_layout_preview":
        raise LayoutReferenceError("layout preview provenance schema/kind mismatch")
    if preview_payload.get("review_status") != "PENDING_HUMAN_REVIEW":
        raise LayoutReferenceError("preview provenance must remain the immutable pre-approval record")
    if preview_payload.get("render_manifest_basis_sha256") != render_manifest_basis_sha256(render_manifest):
        raise LayoutReferenceError("layout preview was made from a different render-manifest basis")
    preview_record = preview_payload.get("preview", {})
    if Path(str(preview_record.get("path", ""))).resolve() != preview:
        raise LayoutReferenceError("preview provenance path mismatch")
    if preview_record.get("sha256") != sha256(preview):
        raise LayoutReferenceError("preview provenance image hash mismatch")
    if (preview_record.get("width"), preview_record.get("height")) != PRODUCTION_SIZE:
        raise LayoutReferenceError("preview provenance dimensions are not 1920x1080")
    with Image.open(preview) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != PRODUCTION_SIZE:
            raise LayoutReferenceError("approved production preview is not a 1920x1080 RGB PNG")
    sheet_record = preview_payload.get("review_sheet", {})
    if Path(str(sheet_record.get("path", ""))).resolve() != review_sheet:
        raise LayoutReferenceError("preview provenance review-sheet path mismatch")
    if sheet_record.get("sha256") != sha256(review_sheet):
        raise LayoutReferenceError("preview provenance review-sheet hash mismatch")
    if (sheet_record.get("width"), sheet_record.get("height")) != REVIEW_SHEET_SIZE:
        raise LayoutReferenceError("preview provenance review-sheet dimensions are not 1920x720")
    with Image.open(review_sheet) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != REVIEW_SHEET_SIZE:
            raise LayoutReferenceError("approved layout review sheet is not a 1920x720 RGB PNG")
    contract = preview_payload.get("quality_contract", {})
    if contract.get("structure_source") != "production _clip_frame path with synthetic waveform/Log-Mel placeholders":
        raise LayoutReferenceError("preview provenance production-renderer contract mismatch")
    if contract.get("required_human_checks") != list(REQUIRED_QUALITY_CHECKS):
        raise LayoutReferenceError("preview provenance quality checklist mismatch")
    if not all(
        contract.get(key) is True
        for key in (
            "reference_is_composition_only",
            "reference_text_people_values_and_density_curves_are_non_authoritative",
            "current_renderer_rules_take_precedence",
        )
    ):
        raise LayoutReferenceError("preview provenance authority contract mismatch")
    expected_sources = {
        str((SKILL_DIR / "scripts" / "render_layout_preview.py").resolve()),
        str((SKILL_DIR / "scripts" / "toolkit" / "rendering" / "render.py").resolve()),
        str((SKILL_DIR / "scripts" / "toolkit" / "rendering" / "common.py").resolve()),
        str(Path(__file__).resolve()),
    }
    source_rows = preview_payload.get("renderer_sources")
    if not isinstance(source_rows, list) or {
        str(row.get("path", "")) for row in source_rows if isinstance(row, dict)
    } != expected_sources:
        raise LayoutReferenceError("preview provenance renderer-source coverage mismatch")
    for row in source_rows:
        source = Path(row["path"])
        if not source.is_file() or row.get("sha256") != sha256(source):
            raise LayoutReferenceError(f"preview renderer source changed after review: {source}")
    reference_record = preview_payload.get("reference", {})
    if reference_record.get("asset", {}).get("sha256") != reference["asset"]["sha256"]:
        raise LayoutReferenceError("preview used a different layout reference asset")
    if reference_record.get("manifest", {}).get("sha256") != reference["manifest"]["sha256"]:
        raise LayoutReferenceError("preview used a different layout reference manifest")
    return {
        "status": "APPROVED",
        "approval_basis": review["approval_basis"].strip(),
        "quality_checks": checks,
        "preview": {"path": str(preview), "sha256": sha256(preview)},
        "preview_provenance": {"path": str(provenance), "sha256": sha256(provenance)},
        "review_sheet": {"path": str(review_sheet), "sha256": sha256(review_sheet)},
        "render_manifest_basis_sha256": render_manifest_basis_sha256(render_manifest),
    }
