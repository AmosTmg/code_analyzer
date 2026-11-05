"""Unit tests for the safe ZIP extraction module."""

import io
import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.extractor import (
    EmptyArchiveError,
    ExtractionError,
    InvalidZipError,
    SecurityError,
    safe_extract_zip,
)


class TestSafeExtractZip(unittest.TestCase):
    """Test suite for safe_extract_zip security and functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.extract_dir = os.path.join(self.temp_dir, "extracted")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_zip_buffer(self, file_map: dict[str, str]) -> io.BytesIO:
        """Helper to create an in-memory ZIP archive from a dict of {filename: content}."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename, content in file_map.items():
                zf.writestr(filename, content)
        buffer.seek(0)
        return buffer

    def test_extract_valid_archive_in_memory(self):
        """Verify normal files and nested subdirectories extract correctly."""
        files = {
            "main.py": "print('hello world')",
            "utils/helpers.py": "def add(a, b): return a + b",
            "docs/readme.txt": "Documentation file.",
        }
        zip_buf = self._create_zip_buffer(files)
        result_path = safe_extract_zip(zip_buf, self.extract_dir)

        self.assertEqual(result_path, Path(self.extract_dir).resolve())
        for rel_path, content in files.items():
            full_path = Path(self.extract_dir) / rel_path
            self.assertTrue(full_path.exists(), f"File {rel_path} was not extracted.")
            self.assertEqual(full_path.read_text(encoding="utf-8"), content)

    def test_extract_from_file_path_on_disk(self):
        """Verify extraction works when passing a path string."""
        zip_path = os.path.join(self.temp_dir, "sample.zip")
        files = {"app.py": "import os"}
        zip_buf = self._create_zip_buffer(files)
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

        result_path = safe_extract_zip(zip_path, self.extract_dir)
        extracted_file = result_path / "app.py"
        self.assertTrue(extracted_file.exists())
        self.assertEqual(extracted_file.read_text(), "import os")

    def test_empty_archive_raises_empty_archive_error(self):
        """Verify an archive with zero entries raises EmptyArchiveError."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            pass  # write nothing
        buffer.seek(0)

        with self.assertRaises(EmptyArchiveError):
            safe_extract_zip(buffer, self.extract_dir)

    def test_archive_with_only_directories_raises_empty_archive_error(self):
        """Verify an archive containing only folders raises EmptyArchiveError."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr("empty_dir/", "")
            zf.writestr("nested/dir/", "")
        buffer.seek(0)

        with self.assertRaises(EmptyArchiveError):
            safe_extract_zip(buffer, self.extract_dir)

    def test_path_traversal_relative_raises_security_error(self):
        """Verify relative path traversal (Zip Slip with '..') is rejected."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr("../evil.txt", "malicious payload")
        buffer.seek(0)

        with self.assertRaises(SecurityError):
            safe_extract_zip(buffer, self.extract_dir)

    def test_path_traversal_nested_relative_raises_security_error(self):
        """Verify nested relative path traversal ('dir/../../evil.txt') is rejected."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr("subfolder/../../evil.txt", "malicious payload")
        buffer.seek(0)

        with self.assertRaises(SecurityError):
            safe_extract_zip(buffer, self.extract_dir)

    def test_path_traversal_absolute_raises_security_error(self):
        """Verify absolute path references are rejected."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr("/tmp/evil.txt", "malicious payload")
        buffer.seek(0)

        with self.assertRaises(SecurityError):
            safe_extract_zip(buffer, self.extract_dir)

    def test_corrupted_or_non_zip_raises_invalid_zip_error(self):
        """Verify non-ZIP data raises InvalidZipError."""
        corrupted_data = io.BytesIO(b"This is definitely not a zip file.")

        with self.assertRaises(InvalidZipError):
            safe_extract_zip(corrupted_data, self.extract_dir)

    def test_symlink_entry_raises_security_error(self):
        """Verify ZIP entries marked as symbolic links are rejected."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            info = zipfile.ZipInfo("symlink_entry")
            # Set POSIX symlink attribute (S_IFLNK)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
        buffer.seek(0)

        with self.assertRaises(SecurityError):
            safe_extract_zip(buffer, self.extract_dir)


if __name__ == "__main__":
    unittest.main()
