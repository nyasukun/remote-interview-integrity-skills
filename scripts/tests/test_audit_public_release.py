from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_public_release.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "audit_public_release_under_test",
    MODULE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load audit module: {MODULE_PATH}")
audit = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = audit
MODULE_SPEC.loader.exec_module(audit)


class ApprovedSyntheticMp4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PurePosixPath("examples/synthetic-interview/test-approved.mp4")
        self.data = b"\x00\x00\x00\x18ftypisom" + b"synthetic-test-data"

    def approved(self, data: bytes | None = None) -> object:
        payload = self.data if data is None else data
        return audit.ApprovedSyntheticMp4(
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def entry(
        self,
        *,
        path: PurePosixPath | None = None,
        mode: str = "100644",
        size: int | None = None,
        data: bytes | None = None,
    ) -> object:
        payload = self.data if data is None else data
        return audit.RepositoryEntry(
            relative=self.path if path is None else path,
            mode=mode,
            size=len(payload) if size is None else size,
            data=payload,
        )

    def findings_for(
        self,
        entry: object,
        approved: object | None = None,
        *,
        path: PurePosixPath | None = None,
    ) -> list[str]:
        approved_map = {}
        if approved is not None:
            approved_map[self.path if path is None else path] = approved
        with mock.patch.dict(
            audit.APPROVED_SYNTHETIC_MP4S,
            approved_map,
            clear=True,
        ):
            return audit.audit_entries([entry])

    def test_published_allowlist_values_are_exact(self) -> None:
        expected = {
            PurePosixPath(
                "examples/synthetic-interview/"
                "deterministic_synthetic_interview_blind.mp4"
            ): (
                "579bc0197f8bb03c0dc1d4250437d2fb7a77c074ec3829d7374718fe2055729c",
                27549459,
            ),
            PurePosixPath(
                "examples/synthetic-interview/closure_evidence_video.mp4"
            ): (
                "5275178ca4dd408b055f4ee9842d2149febbca495ce59d85acee42127173de86",
                21625946,
            ),
            PurePosixPath(
                "examples/synthetic-interview/voice_signal_comparison.mp4"
            ): (
                "2fd5b3305947bc043e214ce6c2b6f8dc0bb7ca543e6a2d1a8a3fc761a580f3f4",
                1337296,
            ),
        }
        actual = {
            path: (approved.sha256, approved.size_bytes)
            for path, approved in audit.APPROVED_SYNTHETIC_MP4S.items()
        }
        self.assertEqual(expected, actual)

    def test_exact_path_mode_size_structure_and_hash_pass(self) -> None:
        findings = self.findings_for(self.entry(), self.approved())
        self.assertEqual([], findings)

    def test_hash_mismatch_fails(self) -> None:
        approved = audit.ApprovedSyntheticMp4(
            sha256="0" * 64,
            size_bytes=len(self.data),
        )
        findings = self.findings_for(self.entry(), approved)
        self.assertIn(
            f"approved MP4 SHA-256 mismatch: {self.path}",
            findings,
        )

    def test_size_and_data_length_mismatch_fail(self) -> None:
        approved = audit.ApprovedSyntheticMp4(
            sha256=hashlib.sha256(self.data).hexdigest(),
            size_bytes=len(self.data) + 1,
        )
        findings = self.findings_for(self.entry(), approved)
        self.assertTrue(
            any(finding.startswith("approved MP4 size mismatch:") for finding in findings)
        )
        self.assertTrue(
            any(
                finding.startswith("approved MP4 data length mismatch:")
                for finding in findings
            )
        )

    def test_missing_data_fails_closed(self) -> None:
        entry = audit.RepositoryEntry(
            relative=self.path,
            mode="100644",
            size=len(self.data),
            data=None,
        )
        findings = self.findings_for(entry, self.approved())
        self.assertIn(f"approved MP4 cannot be validated: {self.path}", findings)

    def test_same_bytes_at_another_path_remain_forbidden(self) -> None:
        other = PurePosixPath("examples/synthetic-interview/not-approved.mp4")
        findings = self.findings_for(
            self.entry(path=other),
            self.approved(),
        )
        self.assertIn(f"forbidden file type: {other}", findings)

    def test_mode_mismatch_fails(self) -> None:
        findings = self.findings_for(
            self.entry(mode="100755"),
            self.approved(),
        )
        self.assertIn(
            f"approved MP4 mode mismatch: {self.path} (expected 100644)",
            findings,
        )

    def test_invalid_ftyp_fails(self) -> None:
        invalid = self.data[:4] + b"nope" + self.data[8:]
        findings = self.findings_for(
            self.entry(data=invalid),
            self.approved(invalid),
        )
        self.assertIn(f"approved MP4 structure is invalid: {self.path}", findings)

    def test_approved_path_in_forbidden_directory_still_fails(self) -> None:
        forbidden_path = PurePosixPath("outputs/test-approved.mp4")
        findings = self.findings_for(
            self.entry(path=forbidden_path),
            self.approved(),
            path=forbidden_path,
        )
        self.assertIn(
            f"private/generated directory: {forbidden_path}",
            findings,
        )

    def test_generic_large_file_limit_is_unchanged(self) -> None:
        path = PurePosixPath("notes/large.txt")
        entry = audit.RepositoryEntry(
            relative=path,
            mode="100644",
            size=audit.MAX_FILE_SIZE + 1,
            data=None,
        )
        findings = self.findings_for(entry)
        self.assertIn(f"file larger than 5 MiB: {path}", findings)

    def test_generic_mp4_is_still_forbidden(self) -> None:
        path = PurePosixPath("examples/synthetic-interview/other.mp4")
        findings = self.findings_for(self.entry(path=path))
        self.assertIn(f"forbidden file type: {path}", findings)

    def test_should_load_only_expected_large_approved_data(self) -> None:
        large_size = audit.MAX_FILE_SIZE + 1
        approved = audit.ApprovedSyntheticMp4(
            sha256="0" * 64,
            size_bytes=large_size,
        )
        with mock.patch.dict(
            audit.APPROVED_SYNTHETIC_MP4S,
            {self.path: approved},
            clear=True,
        ):
            self.assertTrue(audit._should_load_data(self.path, large_size))
            self.assertFalse(audit._should_load_data(self.path, large_size + 1))
            self.assertFalse(
                audit._should_load_data(
                    PurePosixPath("examples/synthetic-interview/other.mp4"),
                    large_size,
                )
            )
            self.assertTrue(audit._should_load_data(self.path, audit.MAX_FILE_SIZE))


class ApprovedSyntheticMp4LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PurePosixPath("examples/synthetic-interview/loader-test.mp4")
        self.data = b"\x00\x00\x00\x18ftypisom" + b"loader-test-data"
        self.approved = audit.ApprovedSyntheticMp4(
            sha256=hashlib.sha256(self.data).hexdigest(),
            size_bytes=len(self.data),
        )

    def test_working_tree_loader_reads_expected_approved_data_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / self.path
            target.parent.mkdir(parents=True)
            target.write_bytes(self.data)

            with (
                mock.patch.object(audit, "MAX_FILE_SIZE", 8),
                mock.patch.dict(
                    audit.APPROVED_SYNTHETIC_MP4S,
                    {self.path: self.approved},
                    clear=True,
                ),
            ):
                entries = audit._working_tree_entries(root)

        matching = [entry for entry in entries if entry.relative == self.path]
        self.assertEqual(1, len(matching))
        self.assertEqual(self.data, matching[0].data)

    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_staged_loader_reads_expected_approved_data_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            target = root / self.path
            target.parent.mkdir(parents=True)
            target.write_bytes(self.data)
            subprocess.run(
                ["git", "add", "-f", "--", self.path.as_posix()],
                cwd=root,
                check=True,
            )

            with (
                mock.patch.object(audit, "MAX_FILE_SIZE", 8),
                mock.patch.dict(
                    audit.APPROVED_SYNTHETIC_MP4S,
                    {self.path: self.approved},
                    clear=True,
                ),
            ):
                entry = audit._staged_entry(root, self.path.as_posix())

        self.assertEqual(self.data, entry.data)


if __name__ == "__main__":
    unittest.main()
