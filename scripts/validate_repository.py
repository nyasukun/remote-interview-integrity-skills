#!/usr/bin/env python3
"""Run the repository's test suites in isolated processes, then audit publication files."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AV_TOOLKIT = ROOT / "skills/interview-av-integrity/scripts/toolkit"
VOICE_TOOLKIT = ROOT / "skills/interview-voice-signal-comparison/scripts/toolkit"


@dataclass(frozen=True)
class TestSuite:
    directory: Path
    import_paths: tuple[Path, ...] = ()


SUITES = {
    "audit": TestSuite(ROOT / "scripts/tests"),
    "av": TestSuite(AV_TOOLKIT / "tests", (AV_TOOLKIT,)),
    "acoustics": TestSuite(VOICE_TOOLKIT / "acoustics/tests", (VOICE_TOOLKIT,)),
    "rendering": TestSuite(
        VOICE_TOOLKIT / "rendering/tests",
        (VOICE_TOOLKIT, VOICE_TOOLKIT / "rendering"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", action="append", choices=SUITES,
        help="run only this test suite (repeatable); the publication audit always runs",
    )
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.suite or SUITES))
    failures: list[str] = []

    # Separate interpreters prevent identically named script modules from leaking
    # between suites. Temp caches keep the audited working tree free of outputs.
    with tempfile.TemporaryDirectory(prefix="interview-skill-validation-") as temporary:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
        environment["MPLBACKEND"] = "Agg"
        environment["XDG_CACHE_HOME"] = str(Path(temporary) / "cache")
        for name in selected:
            suite = SUITES[name]
            suite_environment = environment.copy()
            suite_environment["PYTHONPATH"] = os.pathsep.join(
                str(path) for path in suite.import_paths
            )
            print(f"\nRunning {name} tests", flush=True)
            result = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(suite.directory), "-v"],
                cwd=ROOT, env=suite_environment, check=False,
            )
            if result.returncode:
                failures.append(name)

        print("\nAuditing working-tree publication files", flush=True)
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/audit_public_release.py")],
            cwd=ROOT, env=environment, check=False,
        )
        if result.returncode:
            failures.append("publication audit")

    if failures:
        print(f"\nFAIL: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nPASS: {len(selected)} test suites and publication audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
