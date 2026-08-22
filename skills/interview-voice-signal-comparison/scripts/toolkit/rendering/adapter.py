"""Map the authoritative acoustic metric registry to renderer display fields."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping


TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from acoustics.model import scalar_metrics  # noqa: E402


def display_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = scalar_metrics(summary)
    return {
        "pitch_median_hz": metrics["f0_median_hz"],
        "periodicity_median": metrics["periodicity_median"],
        "active_fraction": metrics["active_frame_fraction"],
        "voiced_fraction": metrics["voiced_fraction_of_active"],
        "rms_dbfs": metrics["rms_dbfs"],
        "band_ratio": {
            "low_0_500": metrics["mean_active_band_fraction_low_0_500"],
            "mid_500_2000": metrics["mean_active_band_fraction_mid_500_2000"],
            "high_2000_8000": metrics["mean_active_band_fraction_high_2000_8000"],
        },
    }
