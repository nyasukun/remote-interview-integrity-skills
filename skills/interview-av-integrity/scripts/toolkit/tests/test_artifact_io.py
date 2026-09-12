from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from video_integrity_analyzer.artifact_io import (
    atomic_text_writer,
    atomic_write_text,
    csv_cell,
    finite_float,
    json_ready,
    load_events_document,
    nested,
    now_utc,
    path_fingerprint,
    visual_config_from_automated,
    write_csv,
    write_json,
)
from video_integrity_analyzer.plosive_sync import VisualReleaseConfig


class _Unserializable:
    pass


class _ExplodingCell:
    def __str__(self) -> str:
        raise RuntimeError("cell rendering failed")


def _temporary_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.suffix == ".tmp")


class JsonReadyTests(unittest.TestCase):
    def test_nonfinite_floats_become_none_at_every_depth(self) -> None:
        converted = json_ready(
            {"median_aperture": math.nan, "values": (1.0, math.inf), "deep": [{"x": -math.inf}]}
        )
        self.assertEqual(
            converted,
            {"median_aperture": None, "values": [1.0, None], "deep": [{"x": None}]},
        )

    def test_numpy_scalars_and_zero_dimensional_arrays_are_unwrapped(self) -> None:
        converted = json_ready(
            {
                "f64": np.float64(1.5),
                "i64": np.int64(3),
                "flag": np.bool_(True),
                "f32_nan": np.float32(math.nan),
                "zero_d": np.array(2.0),
            }
        )
        self.assertEqual(
            converted,
            {"f64": 1.5, "i64": 3, "flag": True, "f32_nan": None, "zero_d": 2.0},
        )
        self.assertIs(type(converted["f64"]), float)
        self.assertIs(type(converted["i64"]), int)
        self.assertIs(type(converted["flag"]), bool)

    def test_mapping_keys_are_coerced_and_unknown_objects_pass_through(self) -> None:
        marker = _Unserializable()
        converted = json_ready({1: "one", "obj": marker})
        self.assertEqual(converted, {"1": "one", "obj": marker})
        self.assertIs(converted["obj"], marker)


class WriteJsonTests(unittest.TestCase):
    def test_writes_strict_indented_utf8_json_with_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "nested" / "value.json"
            write_json(
                target,
                {"label": "日本", "lag": math.nan, "np_nan": np.float64(math.nan), "pair": (1, 2)},
            )
            text = target.read_text(encoding="utf-8")
            self.assertEqual(
                text,
                '{\n  "label": "日本",\n  "lag": null,\n  "np_nan": null,\n  "pair": [\n    1,\n    2\n  ]\n}\n',
            )
            self.assertEqual(json.loads(text)["lag"], None)
            self.assertEqual(_temporary_files(target.parent), [])

    def test_failed_serialization_keeps_previous_artifact_and_removes_temp(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "value.json"
            target.write_text('{"kept": true}\n', encoding="utf-8")
            with self.assertRaises(TypeError):
                write_json(target, {"bad": _Unserializable()})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"kept": true}\n')
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["value.json"])


class AtomicWriteTests(unittest.TestCase):
    def test_replaces_existing_content_via_same_directory_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "report.txt"
            target.write_text("old", encoding="utf-8")
            with atomic_text_writer(target) as output:
                output.write("new\r\nline")
                pending = _temporary_files(target.parent)
                self.assertEqual(len(pending), 1)
                self.assertTrue(pending[0].startswith(".report.txt."))
                self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(target.read_bytes(), b"new\r\nline")
            self.assertEqual(_temporary_files(target.parent), [])

    def test_exception_inside_writer_leaves_target_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "report.txt"
            atomic_write_text(target, "original")
            with self.assertRaisesRegex(RuntimeError, "midway"):
                with atomic_text_writer(target) as output:
                    output.write("partial")
                    raise RuntimeError("midway")
            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["report.txt"])

    def test_missing_target_is_not_created_when_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "deep" / "report.txt"
            with self.assertRaises(RuntimeError):
                with atomic_text_writer(target):
                    raise RuntimeError("abort")
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])


class CsvTests(unittest.TestCase):
    def test_csv_cell_formats_missing_containers_and_scalars(self) -> None:
        self.assertEqual(csv_cell(None), "")
        self.assertEqual(csv_cell(math.nan), "")
        self.assertEqual(csv_cell(np.float64(math.nan)), "")
        self.assertEqual(csv_cell([1, math.nan, "日本"]), '[1,null,"日本"]')
        self.assertEqual(csv_cell({"k": (1, 2)}), '{"k":[1,2]}')
        self.assertEqual(csv_cell(True), "True")
        self.assertEqual(csv_cell(np.float64(2.5)), "2.5")
        self.assertEqual(csv_cell(0.1 + 0.2), "0.30000000000000004")
        self.assertEqual(csv_cell("plain"), "plain")

    def test_write_csv_unions_columns_in_first_seen_order_with_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "table.csv"
            write_csv(
                target,
                [
                    {"event_id": "e1", "lag_ms": 1.5, "reasons": ["a", "b"]},
                    {"event_id": "e2", "extra": np.int64(7), "lag_ms": None},
                ],
            )
            self.assertEqual(
                target.read_bytes(),
                b'event_id,lag_ms,reasons,extra\r\n'
                b'e1,1.5,"[""a"",""b""]",\r\n'
                b'e2,,,7\r\n',
            )
            self.assertEqual(_temporary_files(target.parent), [])

    def test_write_csv_empty_table_is_header_only_unless_placeholder_requested(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / "plain.csv"
            placeholder = Path(folder) / "placeholder.csv"
            write_csv(plain, [])
            write_csv(placeholder, [], empty_placeholder_column="empty")
            self.assertEqual(plain.read_bytes(), b"\r\n")
            self.assertEqual(placeholder.read_bytes(), b'empty\r\n""\r\n')

    def test_write_csv_failure_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "table.csv"
            write_csv(target, [{"a": 1}])
            with self.assertRaisesRegex(RuntimeError, "cell rendering failed"):
                write_csv(target, [{"a": _ExplodingCell()}])
            self.assertEqual(target.read_bytes(), b"a\r\n1\r\n")
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["table.csv"])


class PathFingerprintTests(unittest.TestCase):
    def test_fingerprint_reports_resolved_path_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "input.bin"
            payload = b"\x00\x01plosive" * 1000
            target.write_bytes(payload)
            fingerprint = path_fingerprint(target)
            self.assertEqual(
                set(fingerprint), {"path", "size_bytes", "mtime_ns", "sha256"}
            )
            self.assertEqual(fingerprint["path"], str(target.resolve()))
            self.assertEqual(fingerprint["size_bytes"], len(payload))
            self.assertEqual(fingerprint["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(fingerprint["mtime_ns"], target.stat().st_mtime_ns)


class LoadEventsDocumentTests(unittest.TestCase):
    def test_default_accepts_only_the_events_key(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "doc.json"
            path.write_text(json.dumps({"events": [{"id": 1}], "meta": 1}), encoding="utf-8")
            loaded = load_events_document(path, "doc")
            self.assertEqual(loaded, {"events": [{"id": 1}], "meta": 1})
            path.write_text(json.dumps({"annotations": [{"id": 1}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "doc must contain a top-level events list"):
                load_events_document(path, "doc")

    def test_alias_keys_are_normalized_into_events(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "doc.json"
            path.write_text(
                json.dumps({"events": "not a list", "annotations": [{"id": 2}]}),
                encoding="utf-8",
            )
            loaded = load_events_document(
                path, "annotations JSON", list_keys=("events", "annotations")
            )
            self.assertEqual(loaded["events"], [{"id": 2}])
            self.assertEqual(loaded["annotations"], [{"id": 2}])

    def test_rejects_non_object_documents(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "doc.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level JSON object"):
                load_events_document(path, "doc")


class SupportHelperTests(unittest.TestCase):
    def test_nested_returns_empty_mapping_for_missing_or_non_mapping(self) -> None:
        self.assertEqual(nested({"a": {"b": 1}}, "a"), {"b": 1})
        self.assertEqual(nested({"a": None}, "a"), {})
        self.assertEqual(nested({"a": [1]}, "a"), {})
        self.assertEqual(nested(None, "a"), {})
        self.assertEqual(nested("text", "a"), {})

    def test_finite_float_filters_missing_and_nonfinite(self) -> None:
        self.assertIsNone(finite_float(None))
        self.assertIsNone(finite_float("abc"))
        self.assertIsNone(finite_float(math.nan))
        self.assertIsNone(finite_float(math.inf))
        self.assertEqual(finite_float("1.5"), 1.5)
        self.assertEqual(finite_float(np.float64(2.0)), 2.0)

    def test_visual_config_from_automated_applies_known_keys_only(self) -> None:
        self.assertEqual(visual_config_from_automated({}), VisualReleaseConfig())
        automated = {
            "configuration": {
                "protocol": {
                    "visual_release_config": {
                        "closed_median_max": 0.02,
                        "unknown_future_gate": 1.0,
                    }
                }
            }
        }
        config = visual_config_from_automated(automated)
        self.assertEqual(config.closed_median_max, 0.02)
        self.assertEqual(config.reopened_median_min, VisualReleaseConfig().reopened_median_min)

    def test_now_utc_is_timezone_aware_iso8601(self) -> None:
        stamp = datetime.fromisoformat(now_utc())
        self.assertEqual(stamp.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()
