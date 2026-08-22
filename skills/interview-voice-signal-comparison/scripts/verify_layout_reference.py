#!/usr/bin/env python3
"""Verify the approved voice-comparison layout reference and print provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path


RENDERING_DIR = Path(__file__).resolve().parent / "toolkit" / "rendering"
sys.path.insert(0, str(RENDERING_DIR))

from layout_reference import verify_layout_reference  # noqa: E402


def main() -> int:
    try:
        result = verify_layout_reference()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
