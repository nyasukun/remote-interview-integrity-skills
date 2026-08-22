#!/usr/bin/env python3
"""Portable entry point for comparison-video and mapping QA."""

from __future__ import annotations

import sys
from pathlib import Path


RENDERING = Path(__file__).resolve().parent / "toolkit" / "rendering"
sys.path.insert(0, str(RENDERING))

from verify import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
