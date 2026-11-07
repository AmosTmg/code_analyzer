"""Unit tests for the repository file filtering and language detection module."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.filter import (
    EXTENSION_TO_LANGUAGE,
    filter_repository,
    get_language,
    is_binary_file,
    is_ignored_directory,
    is_ignored_file,
)


class TestFilterModule(unittest.TestCase):
    """Test suite for repository scanning, filtering, and language classification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extension_to_language_mapping(self):
        """Verify all required extensions map to their correct language."""
        expected_mappings = {
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".json": "json",
            ".md": "markdown",
        }
        for ext, expected_lang in expected_mappings.items():
            self.assertEqual(
                get_language(f"test{ext}"),
                expected_lang,
                f"Failed mapping for extension {ext}",
            )
            # Verify case insensitivity
            self.assertEqual(
                get_language(f"TEST{ext.upper()}"),
                expected_lang,
                f"Failed case-insensitive mapping for {ext}",
            )

    def test_unsupported_extensions_return_none(self):
        """Verify unsupported extensions are rejected."""
        unsupported = [
            "image.png",
            "archive.zip",
            "script.sh",
            "document.pdf",
            "binary.exe",
            "gemfile.rb",
        ]
        for name in unsupported:
            self.assertIsNone(get_language(name))

    def test_ignored_directories(self):
        """Verify standard ignored directories are detected."""
        ignored = [
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "__pycache__",
            "dist",
            "build",
            ".idea",
        ]
        for d in ignored:
            self.assertTrue(is_ignored_directory(d), f"Directory {d} should be ignored")
            self.assertTrue(is_ignored_directory(d.upper()), f"Directory {d.upper()} should be ignored")

        allowed = [".github", ".gitlab", "src", "app", "lib", "components", "utils", "tests"]
        for d in allowed:
            self.assertFalse(is_ignored_directory(d), f"Directory {d} should NOT be ignored")

    def test_ignored_files(self):
        """Verify lockfiles, minified files, and hidden files are ignored."""
        ignored = [
            ".env",
            ".DS_Store",
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            "app.min.js",
            "styles.min.css",
            "bundle.bundle.js",
        ]
        for f in ignored:
            self.assertTrue(is_ignored_file(f), f"File {f} should be ignored")

        allowed = ["main.py", "package.json", "index.html", "query.sql"]
        for f in allowed:
            self.assertFalse(is_ignored_file(f), f"File {f} should NOT be ignored")

    def test_binary_detection(self):
        """Verify binary files with null bytes or unreadable bytes are detected."""
        text_file = Path(self.temp_dir) / "normal.py"
        text_file.write_text("print('hello')", encoding="utf-8")
        self.assertFalse(is_binary_file(text_file))

        empty_file = Path(self.temp_dir) / "empty.py"
        empty_file.write_bytes(b"")
        self.assertFalse(is_binary_file(empty_file))

        binary_file = Path(self.temp_dir) / "fake.py"
        binary_file.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00")
        self.assertTrue(is_binary_file(binary_file))

    def test_empty_repository(self):
        """Verify an empty directory returns an empty file list."""
        results = filter_repository(self.temp_dir)
        self.assertEqual(results, [])

    def test_repository_with_only_ignored_directories(self):
        """Verify repositories containing only ignored directories yield no files."""
        node_modules = Path(self.temp_dir) / "node_modules"
        node_modules.mkdir()
        (node_modules / "index.js").write_text("console.log('ignored');", encoding="utf-8")

        pycache = Path(self.temp_dir) / "__pycache__"
        pycache.mkdir()
        (pycache / "module.cpython-314.py").write_text("cache", encoding="utf-8")

        results = filter_repository(self.temp_dir)
        self.assertEqual(results, [])

    def test_nested_repository_filtering(self):
        """Verify end-to-end filtering across nested folders, ignoring bloat."""
        root = Path(self.temp_dir)

        # Valid source files in nested directories
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("def run(): pass", encoding="utf-8")

        (root / "src" / "components").mkdir()
        (root / "src" / "components" / "Button.tsx").write_text("export const Button = () => null;", encoding="utf-8")
        (root / "src" / "components" / "Button.css").write_text(".btn { color: red; }", encoding="utf-8")

        (root / "README.md").write_text("# Project Docs", encoding="utf-8")
        (root / "package.json").write_text('{"name": "test"}', encoding="utf-8")
        (root / "db.sql").write_text("SELECT 1;", encoding="utf-8")

        # Files and directories that should be skipped
        (root / "node_modules").mkdir()
        (root / "node_modules" / "library.js").write_text("module.exports = {};", encoding="utf-8")

        (root / "dist").mkdir()
        (root / "dist" / "bundle.js").write_text("bundled code", encoding="utf-8")

        (root / "package-lock.json").write_text('{"lockfileVersion": 3}', encoding="utf-8")
        (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (root / "app.min.js").write_text("minified", encoding="utf-8")

        results = filter_repository(root)

        # Expected relative paths
        expected_paths = [
            "README.md",
            "db.sql",
            "package.json",
            "src/components/Button.css",
            "src/components/Button.tsx",
            "src/main.py",
        ]

        actual_paths = [rf.relative_path for rf in results]
        self.assertEqual(actual_paths, expected_paths)

        # Verify language mappings for returned items
        path_to_lang = {rf.relative_path: rf.language for rf in results}
        self.assertEqual(path_to_lang["src/main.py"], "python")
        self.assertEqual(path_to_lang["src/components/Button.tsx"], "typescript")
        self.assertEqual(path_to_lang["src/components/Button.css"], "css")
        self.assertEqual(path_to_lang["README.md"], "markdown")
        self.assertEqual(path_to_lang["package.json"], "json")
        self.assertEqual(path_to_lang["db.sql"], "sql")

    def test_filter_repository_invalid_paths(self):
        """Verify errors when pointing to a non-existent path or non-directory."""
        with self.assertRaises(FileNotFoundError):
            filter_repository(Path(self.temp_dir) / "non_existent")

        file_path = Path(self.temp_dir) / "a_file.txt"
        file_path.write_text("content", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            filter_repository(file_path)

    def test_github_and_gitlab_directories_discoverable(self):
        """Verify that directories like .github and .gitlab are scanned and supported files indexed."""
        root = Path(self.temp_dir)
        gh_dir = root / ".github" / "workflows"
        gh_dir.mkdir(parents=True)
        (gh_dir / "ci.md").write_text("# CI Workflow", encoding="utf-8")

        gitlab_dir = root / ".gitlab"
        gitlab_dir.mkdir(parents=True)
        (gitlab_dir / "ci.py").write_text("print('gitlab')", encoding="utf-8")

        results = filter_repository(root)
        paths = [rf.relative_path for rf in results]
        self.assertIn(".github/workflows/ci.md", paths)
        self.assertIn(".gitlab/ci.py", paths)

    def test_file_symlink_excluded(self):
        """Verify that symbolic links to files are not indexed."""
        external_file = Path(self.temp_dir) / "external.py"
        external_file.write_text("print('external')", encoding="utf-8")

        repo_dir = Path(self.temp_dir) / "repo"
        repo_dir.mkdir()
        internal_file = repo_dir / "valid.py"
        internal_file.write_text("print('valid')", encoding="utf-8")

        symlink_file = repo_dir / "linked.py"
        try:
            symlink_file.symlink_to(external_file)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks not supported on this platform/filesystem: {exc}")

        results = filter_repository(repo_dir)
        paths = [rf.relative_path for rf in results]
        self.assertIn("valid.py", paths)
        self.assertNotIn("linked.py", paths)

    def test_directory_symlink_not_traversed(self):
        """Verify that symbolic links to directories are not traversed."""
        external_dir = Path(self.temp_dir) / "external_folder"
        external_dir.mkdir()
        (external_dir / "external_code.py").write_text("print('outside')", encoding="utf-8")

        repo_dir = Path(self.temp_dir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "inside.py").write_text("print('inside')", encoding="utf-8")

        symlink_dir = repo_dir / "linked_dir"
        try:
            symlink_dir.symlink_to(external_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Directory symlinks not supported on this platform/filesystem: {exc}")

        results = filter_repository(repo_dir)
        paths = [rf.relative_path for rf in results]
        self.assertIn("inside.py", paths)
        for p in paths:
            self.assertFalse(p.startswith("linked_dir"), f"Symlinked directory path found: {p}")


if __name__ == "__main__":
    unittest.main()
