#!/usr/bin/env python3
"""Fail-closed runtime preflight for the voice-signal comparison toolkit."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_MODULES = {
    "av": "av",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "Pillow": "PIL",
    "scipy": "scipy",
}


def _module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def check() -> dict[str, Any]:
    failures: list[str] = []
    modules: dict[str, dict[str, str]] = {}

    matplotlib_cache = Path(tempfile.gettempdir()) / "voice-signal-comparison-mpl"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    if sys.version_info < (3, 11):
        failures.append("Python 3.11 or newer is required")

    loaded: dict[str, Any] = {}
    for package, import_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(import_name)
            loaded[package] = module
            modules[package] = {
                "import": import_name,
                "status": "PASS",
                "version": _module_version(module),
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            failures.append(f"cannot import {package}: {type(exc).__name__}: {exc}")
            modules[package] = {
                "import": import_name,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }

    codecs: dict[str, dict[str, str]] = {}
    if "av" in loaded:
        av = loaded["av"]
        for codec_name, mode in (("libx264", "w"), ("aac", "w")):
            try:
                codec = av.codec.Codec(codec_name, mode)
                codecs[codec_name] = {
                    "status": "PASS",
                    "canonical_name": str(codec.name),
                }
            except Exception as exc:  # pragma: no cover - build dependent
                failures.append(
                    f"PyAV encoder unavailable ({codec_name}): "
                    f"{type(exc).__name__}: {exc}"
                )
                codecs[codec_name] = {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "modules": modules,
        "encoders": codecs,
        "network_required": False,
        "failures": failures,
    }


def main() -> int:
    report = check()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
