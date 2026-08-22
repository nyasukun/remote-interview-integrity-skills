#!/usr/bin/env python3
"""Check the local runtime without installing or downloading anything."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


def module_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    toolkit_root = Path(__file__).resolve().parent / "toolkit"
    # Always inspect the implementation shipped with this skill.  A similarly
    # named editable project in the caller's working directory must not shadow
    # the locally installed package during the runtime preflight.
    sys.path.insert(0, str(toolkit_root))
    local_model = toolkit_root / "models" / "face_landmarker.task"
    parser.add_argument("--face-model", type=Path, default=local_model)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    modules = {name: module_status(name) for name in ("av", "numpy", "PIL", "mediapipe")}
    codecs: dict[str, object] = {}
    if modules["av"]["available"]:
        import av

        for codec in ("libx264", "h264", "aac"):
            try:
                av.CodecContext.create(codec, "w")
                codecs[codec] = True
            except Exception as error:
                codecs[codec] = f"{type(error).__name__}: {error}"

    model = args.face_model.expanduser().resolve()
    face_runtime: dict[str, object] = {
        "status": "NOT_RUN",
        "input": "one deterministic synthetic frame; no case media",
        "frame_transport": os.environ.get(
            "VIDEO_INTEGRITY_FACE_TRANSPORT", "auto"
        ),
    }
    if (
        modules["mediapipe"]["available"]
        and modules["numpy"]["available"]
        and model.is_file()
    ):
        try:
            from video_integrity_analyzer.face_process import preflight_face_runtime

            with tempfile.TemporaryDirectory(
                prefix="interview_av_face_preflight_"
            ) as temporary:
                diagnostics = preflight_face_runtime(
                    model_path=model,
                    output_dir=Path(temporary),
                )
            face_runtime.update(status="PASS", diagnostics=diagnostics)
        except Exception as error:
            face_runtime.update(
                status="FAIL",
                error_type=type(error).__name__,
                error=str(error),
            )
    result = {
        "python": sys.version,
        "python_supported": (3, 10) <= sys.version_info[:2] < (3, 13),
        "platform": platform.platform(),
        "modules": modules,
        "encoders": codecs,
        "face_model": {
            "path": str(model),
            "exists": model.is_file(),
        },
        "face_runtime": face_runtime,
    }
    passed = bool(
        result["python_supported"]
        and all(item["available"] for item in modules.values())
        and any(codecs.get(name) is True for name in ("libx264", "h264"))
        and codecs.get("aac") is True
        and model.is_file()
        and face_runtime["status"] == "PASS"
    )
    result["status"] = "PASS" if passed else "FAIL"
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
