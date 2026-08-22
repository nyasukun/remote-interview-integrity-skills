from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--shared-memory")
    transport.add_argument("--frame-file", type=Path)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-faces", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--roi")
    return parser


def _send(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parsed = tuple(float(part) for part in value.split(","))
    if len(parsed) != 4:
        raise ValueError("invalid ROI")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shared: shared_memory.SharedMemory | None = None
    mapped: np.memmap | None = None
    if args.shared_memory:
        shared = shared_memory.SharedMemory(name=args.shared_memory, create=False)
        # The parent owns/unlinks the segment. Avoid a second resource tracker
        # in this short-lived helper trying to unlink the same name at exit.
        try:
            resource_tracker.unregister(shared._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        frame_view = np.ndarray(
            (args.height, args.width, 3), dtype=np.uint8, buffer=shared.buf
        )
    else:
        mapped = np.memmap(
            args.frame_file,
            dtype=np.uint8,
            mode="r",
            shape=(args.height, args.width, 3),
        )
        frame_view = mapped
    try:
        # Import here so this process loads MediaPipe/OpenCV but never PyAV.
        from .face import FaceAnalyzer

        with FaceAnalyzer(
            args.model,
            max_faces=args.max_faces,
            roi=_parse_roi(args.roi),
        ) as analyzer:
            _send({"status": "ready"})
            for raw_line in sys.stdin:
                message = json.loads(raw_line)
                command = message.get("command")
                if command == "frame":
                    assignments = analyzer.process_frame(
                        float(message["time_s"]), frame_view
                    )
                    _send(
                        {
                            "status": "frame",
                            "detections": len(assignments),
                        }
                    )
                elif command == "finish":
                    tracks = analyzer.finish()
                    args.result.parent.mkdir(parents=True, exist_ok=True)
                    with args.result.open("wb") as handle:
                        pickle.dump(
                            {
                                "tracks": tracks,
                                "scene_cut_count": analyzer.scene_cut_count,
                            },
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    _send(
                        {
                            "status": "finished",
                            "tracks": len(tracks),
                            "scene_cuts": analyzer.scene_cut_count,
                        }
                    )
                    return 0
                else:
                    raise ValueError(f"unknown worker command: {command!r}")
        raise RuntimeError("worker input ended before finish command")
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        _send({"status": "error", "error": f"{type(error).__name__}: {error}"})
        return 1
    finally:
        if mapped is not None:
            del mapped
        if shared is not None:
            shared.close()


if __name__ == "__main__":
    raise SystemExit(main())
