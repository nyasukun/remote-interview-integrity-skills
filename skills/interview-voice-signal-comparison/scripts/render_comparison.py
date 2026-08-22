#!/usr/bin/env python3
"""Portable entry point for the manifest-driven comparison renderer."""

from __future__ import annotations

import sys
from pathlib import Path


RENDERING = Path(__file__).resolve().parent / "toolkit" / "rendering"
sys.path.insert(0, str(RENDERING))

from render import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
