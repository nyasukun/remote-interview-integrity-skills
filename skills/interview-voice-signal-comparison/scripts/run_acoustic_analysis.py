#!/usr/bin/env python3
"""Portable entry point for the non-biometric acoustic workflow."""

from __future__ import annotations

import sys
from pathlib import Path


TOOLKIT = Path(__file__).resolve().parent / "toolkit"
sys.path.insert(0, str(TOOLKIT))

from acoustics.run_analysis import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
