"""Integration tests for the repository processing pipeline and Streamlit UI."""

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import (
    _derive_folder_repo_name,
    _derive_zip_repo_name,
    classify_and_process_upload,
    compute_files_upload_key,
    compute_folder_upload_key,
    compute_unified_upload_key,
    compute_upload_key,
    parse_unified_upload,
    process_repository_archive,
    process_repository_files,
    process_repository_folder,
    safe_write_folder_files,
)
from src.extractor import EmptyArchiveError, SecurityError
from src.models import RepoSummary
from src.retriever import TFIDFRetriever


class MockUploadedFile:
    """Mock simulating Streamlit UploadedFile with name and content."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self._buffer = io.BytesIO(content)

    def getvalue(self) -> bytes:
        return self._content

    def read(self, *args, **kwargs) -> bytes:
        return self._buffer.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._buffer.seek(*args, **kwargs)


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end integration tests for ZIP and folder ingestion and retrieval pipelines."""

    def _create_sample_repo_zip(self) -> io.BytesIO:
        """Creates an in-memory ZIP archive mimicking a realistic repository."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Python auth module
            zf.writestr(
                "my-repo/src/auth.py",
                "def authenticate_user(username, password):\n"
                "    '''Verify user credentials in database.'''\n"
                "    return db.check_credentials(username, password)\n",
            )
            # Python database module
            zf.writestr(
                "my-repo/src/db.py",
                "def connect_database(host, port, db_name):\n"
                "    '''Create connection to PostgreSQL database.'''\n"
                "    return create_connection(host, port, db_name)\n",
            )
            # JavaScript client
            zf.writestr(
                "my-repo/static/client.js",
                "function renderDashboard(userId) {\n"
                "    console.log('Rendering dashboard for user ' + userId);\n"
                "}\n",
            )
            # Documentation
            zf.writestr(
                "my-repo/README.md",
                "# Sample Repository\nThis repository provides auth and database utilities.\n",
            )
            # Ignored bloat (should not be indexed)
            zf.writestr("my-repo/.git/config", "dummy git config")
            zf.writestr("my-repo/node_modules/pkg/index.js", "ignored js")
            zf.writestr("my-repo/package-lock.json", "{}")
        buffer.seek(0)
        return buffer

    def test_process_repository_archive_success(self):
        """Verify full pipeline: extract -> filter -> chunk -> index -> retrieve."""
        zip_buf = self._create_sample_repo_zip()
        retriever, summary = process_repository_archive(
            zip_source=zip_buf,
            repo_name="my-repo",
            chunk_size=50,
            overlap=10,
        )

        self.assertIsInstance(retriever, TFIDFRetriever)
        self.assertIsInstance(summary, RepoSummary)
        self.assertEqual(summary.repo_name, "my-repo")

        # 4 supported files: src/auth.py, src/db.py, static/client.js, README.md
        self.assertEqual(summary.total_files, 4)
        self.assertGreaterEqual(summary.total_chunks, 4)
        self.assertEqual(summary.language_breakdown.get("python"), 2)
        self.assertEqual(summary.language_breakdown.get("javascript"), 1)
        self.assertEqual(summary.language_breakdown.get("markdown"), 1)

        # Search against the resulting retriever
        results = retriever.search("database credentials authentication", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn(top_match.chunk.file_path, ["my-repo/src/auth.py", "my-repo/src/db.py"])
        self.assertGreater(top_match.score, 0.0)

    def test_process_repository_no_supported_files_raises_value_error(self):
        """Verify repositories without supported files raise a helpful ValueError."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr("binary.exe", b"\x7fELF\x00")
            zf.writestr("image.png", b"\x89PNG\r\n")
        buffer.seek(0)

        with self.assertRaises(ValueError) as ctx:
            process_repository_archive(buffer, "binary-repo")
        self.assertIn("No supported source code files", str(ctx.exception))

    def test_process_repository_empty_archive_raises_empty_archive_error(self):
        """Verify empty ZIP archives raise EmptyArchiveError."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            pass
        buffer.seek(0)

        with self.assertRaises(EmptyArchiveError):
            process_repository_archive(buffer, "empty-repo")

    def test_process_repository_path_traversal_raises_security_error(self):
        """Verify malicious archives raise SecurityError."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr("../evil.py", "print('evil')")
        buffer.seek(0)

        with self.assertRaises(SecurityError):
            process_repository_archive(buffer, "evil-repo")

    def test_compute_upload_key_differentiates_same_size_different_content(self):
        """Verify upload key uses SHA-256 so archives of identical size have unique keys."""
        payload_a = b"content_alpha_123456"
        payload_b = b"content_bravo_123456"
        self.assertEqual(len(payload_a), len(payload_b))

        key_a = compute_upload_key("repo.zip", payload_a)
        key_b = compute_upload_key("repo.zip", payload_b)

        self.assertNotEqual(key_a, key_b)
        self.assertTrue(key_a.startswith("zip_repo.zip_"))
        self.assertTrue(key_b.startswith("zip_repo.zip_"))

        # Renaming the same content updates the key
        key_renamed = compute_upload_key("renamed.zip", payload_a)
        self.assertNotEqual(key_a, key_renamed)
        self.assertTrue(key_renamed.startswith("zip_renamed.zip_"))

    # --- Folder Upload Integration Tests ---

    def test_process_repository_folder_success(self):
        """Verify direct folder upload preserves nested relative paths and builds working index."""
        files = [
            MockUploadedFile(
                "my_app/src/auth.py",
                b"def login(user, pwd):\n    return verify(user, pwd)\n",
            ),
            MockUploadedFile(
                "my_app/src/utils/calc.py",
                b"def compute_total(items):\n    return sum(items)\n",
            ),
            MockUploadedFile(
                "my_app/README.md",
                b"# My Application\nWelcome to my app documentation.\n",
            ),
            MockUploadedFile(
                "my_app/node_modules/lib.js",
                b"console.log('ignored');",
            ),
        ]

        retriever, summary = process_repository_folder(files, repo_name="my_app")

        self.assertIsInstance(retriever, TFIDFRetriever)
        self.assertIsInstance(summary, RepoSummary)
        self.assertEqual(summary.repo_name, "my_app")

        # 3 supported files (node_modules is pruned)
        self.assertEqual(summary.total_files, 3)
        self.assertIn("my_app/src/auth.py", summary.file_list)
        self.assertIn("my_app/src/utils/calc.py", summary.file_list)
        self.assertIn("my_app/README.md", summary.file_list)
        self.assertNotIn("my_app/node_modules/lib.js", summary.file_list)

        # Search against folder-indexed repository
        results = retriever.search("verify user login", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.file_path, "my_app/src/auth.py")
        self.assertGreater(results[0].score, 0.0)

    def test_folder_upload_empty_raises_value_error(self):
        """Verify empty folder upload raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            process_repository_folder([], "empty_folder")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_folder_upload_parent_traversal_raises_security_error(self):
        """Verify parent traversal (..) in folder upload paths is rejected."""
        files = [MockUploadedFile("../escape.py", b"evil code")]
        with self.assertRaises(SecurityError):
            process_repository_folder(files, "traversal_repo")

        files_nested = [MockUploadedFile("dir/../../escape.py", b"evil code")]
        with self.assertRaises(SecurityError):
            process_repository_folder(files_nested, "traversal_repo")

    def test_folder_upload_absolute_path_raises_security_error(self):
        """Verify absolute paths in folder upload are rejected."""
        files_posix = [MockUploadedFile("/etc/passwd.py", b"evil code")]
        with self.assertRaises(SecurityError):
            process_repository_folder(files_posix, "abs_repo")

        files_win = [MockUploadedFile("\\Windows\\System32.py", b"evil code")]
        with self.assertRaises(SecurityError):
            process_repository_folder(files_win, "abs_repo")

    def test_folder_upload_windows_drive_and_colons_raises_security_error(self):
        """Verify Windows drive letters and alternate data streams are rejected."""
        files_drive = [MockUploadedFile("C:\\Users\\admin.py", b"evil code")]
        with self.assertRaises(SecurityError):
            process_repository_folder(files_drive, "drive_repo")

        files_stream = [MockUploadedFile("file.py:stream", b"evil code")]
        with self.assertRaises(SecurityError):
            process_repository_folder(files_stream, "drive_repo")

    def test_folder_upload_null_byte_raises_security_error(self):
        """Verify null bytes in file paths are rejected."""
        files_null = [MockUploadedFile("src/main\0.py", b"evil code")]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files_null, "null_repo")
        self.assertIn("null byte", str(ctx.exception).lower())

    def test_folder_upload_exact_duplicate_paths_raises_security_error(self):
        """Verify exact duplicate paths in a folder upload are rejected with SecurityError."""
        files = [
            MockUploadedFile("src/main.py", b"def first(): pass\n"),
            MockUploadedFile("src/main.py", b"def second(): pass\n"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "dup_repo")
        self.assertIn("duplicate relative path", str(ctx.exception).lower())

    def test_folder_upload_normalized_duplicate_paths_raises_security_error(self):
        """Verify differing representations of the same normalized path are rejected."""
        # 'src/main.py' and 'src/./main.py'
        files_dot = [
            MockUploadedFile("src/main.py", b"def original(): pass\n"),
            MockUploadedFile("src/./main.py", b"def duplicate(): pass\n"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files_dot, "dup_dot_repo")
        self.assertIn("duplicate relative path", str(ctx.exception).lower())

        # 'src/utils/main.py' and 'src\\utils\\main.py'
        files_backslash = [
            MockUploadedFile("src/utils/main.py", b"def original(): pass\n"),
            MockUploadedFile("src\\utils\\main.py", b"def duplicate(): pass\n"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files_backslash, "dup_slash_repo")
        self.assertIn("duplicate relative path", str(ctx.exception).lower())

    # --- Traversal-before-normalization tests ---

    def test_traversal_via_embedded_dotdot_single_raises_security_error(self):
        """Verify 'src/../evil.py' is caught by raw component check before normalization."""
        files = [MockUploadedFile("src/../evil.py", b"malicious code")]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "traversal_repo")
        self.assertIn("..", str(ctx.exception))

    def test_traversal_via_embedded_dotdot_double_raises_security_error(self):
        """Verify 'src/../../evil.py' is caught by raw component check before normalization."""
        files = [MockUploadedFile("src/../../evil.py", b"malicious code")]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "traversal_repo")
        self.assertIn("..", str(ctx.exception))

    def test_dotdot_prefix_filename_is_accepted(self):
        """Verify a legitimate filename starting with '..' such as '..config.py' is not rejected."""
        with tempfile.TemporaryDirectory() as dest:
            files = [MockUploadedFile("src/..config.py", b"# legitimate dotdot-prefixed file\n")]
            result_dir = safe_write_folder_files(files, dest)
            written = result_dir / "src" / "..config.py"
            self.assertTrue(written.exists(), "Legitimate '..config.py' should be written to disk.")
            self.assertEqual(written.read_bytes(), b"# legitimate dotdot-prefixed file\n")

    # --- File/directory collision detection tests ---

    def test_file_then_directory_collision_raises_security_error(self):
        """Verify 'src' as a file followed by 'src/main.py' raises SecurityError (file→dir collision)."""
        files = [
            MockUploadedFile("src", b"i am a file"),
            MockUploadedFile("src/main.py", b"def foo(): pass\n"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "collision_repo")
        self.assertIn("conflict", str(ctx.exception).lower())

    def test_directory_then_file_collision_raises_security_error(self):
        """Verify 'src/main.py' followed by 'src' as a file raises SecurityError (dir→file collision)."""
        files = [
            MockUploadedFile("src/main.py", b"def foo(): pass\n"),
            MockUploadedFile("src", b"i am a file"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "collision_repo")
        self.assertIn("conflict", str(ctx.exception).lower())

    def test_file_as_directory_then_child_collision_raises_security_error(self):
        """Verify 'src/main.py' as a file then 'src/main.py/utils.py' raises SecurityError."""
        files = [
            MockUploadedFile("src/main.py", b"def foo(): pass\n"),
            MockUploadedFile("src/main.py/utils.py", b"def bar(): pass\n"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "collision_repo")
        self.assertIn("conflict", str(ctx.exception).lower())

    def test_child_then_file_as_directory_collision_raises_security_error(self):
        """Verify 'src/main.py/utils.py' then 'src/main.py' as a file raises SecurityError."""
        files = [
            MockUploadedFile("src/main.py/utils.py", b"def bar(): pass\n"),
            MockUploadedFile("src/main.py", b"def foo(): pass\n"),
        ]
        with self.assertRaises(SecurityError) as ctx:
            process_repository_folder(files, "collision_repo")
        self.assertIn("conflict", str(ctx.exception).lower())

    def test_folder_repo_name_not_dependent_on_upload_order(self):
        """Verify repository display name comes from explicit argument, not file upload order."""
        files_order_1 = [
            MockUploadedFile("README.md", b"# Readme\n"),
            MockUploadedFile("src/index.js", b"console.log('hi');\n"),
        ]
        files_order_2 = [
            MockUploadedFile("src/index.js", b"console.log('hi');\n"),
            MockUploadedFile("README.md", b"# Readme\n"),
        ]

        _, summary_1 = process_repository_folder(files_order_1, repo_name="custom_display_name")
        _, summary_2 = process_repository_folder(files_order_2, repo_name="custom_display_name")

        self.assertEqual(summary_1.repo_name, "custom_display_name")
        self.assertEqual(summary_2.repo_name, "custom_display_name")

    def test_folder_upload_key_deterministic_and_unique(self):
        """Verify folder upload hash properties: order independence, content sensitivity, path sensitivity."""
        item1 = ("src/b.py", b"content B")
        item2 = ("src/a.py", b"content A")

        # 1. Identical contents in different order produce identical hash
        key1 = compute_folder_upload_key("proj", [item1, item2])
        key2 = compute_folder_upload_key("proj", [item2, item1])
        self.assertEqual(key1, key2)
        self.assertTrue(key1.startswith("folder_"))

        # 2. Content change produces different hash
        item2_modified = ("src/a.py", b"content A MODIFIED")
        key_modified = compute_folder_upload_key("proj", [item1, item2_modified])
        self.assertNotEqual(key1, key_modified)

        # 3. Path change produces different hash
        item2_renamed = ("src/a_new.py", b"content A")
        key_renamed = compute_folder_upload_key("proj", [item1, item2_renamed])
        self.assertNotEqual(key1, key_renamed)

        # 4. Folder name change does not alter content hash (display-name independence)
        key_other_folder = compute_folder_upload_key("other_proj", [item1, item2])
        self.assertEqual(key1, key_other_folder)

        # 5. Folder key vs ZIP key are distinct
        zip_key = compute_upload_key("proj.zip", b"some bytes")
        self.assertTrue(zip_key.startswith("zip_"))
        self.assertTrue(key1.startswith("folder_"))
        self.assertNotEqual(zip_key, key1)

    def test_hash_serialization_ambiguity_regression(self):
        """Regression test: verify length-prefixed serialization prevents delimiter ambiguity collisions."""
        # Without length-prefixing, a delimiter like \0 could allow boundaries to shift:
        # Collection A: path 'a', content b'\0c'
        # Collection B: path 'a\0', content b'c'
        item_a = ("a", b"\0c")
        item_b = ("a\0", b"c")

        key_a = compute_folder_upload_key("repo", [item_a])
        key_b = compute_folder_upload_key("repo", [item_b])

        self.assertNotEqual(
            key_a,
            key_b,
            "Length-prefixed serialization must differentiate paths and content without delimiter ambiguity.",
        )

    # --- Streamlit UI Render Tests ---

    def test_app_initial_render_via_streamlit_apptest(self):
        """Verify that app.py initializes cleanly using Streamlit's official AppTest harness."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run()

        # Verify page title and header
        self.assertFalse(at.exception, f"App crashed on startup: {at.exception}")
        self.assertIn("Context-Aware Codebase Analyzer", at.title[0].value)

        # Unified uploader: exactly 0 separate file_uploader boxes, 0 radio selectors
        self.assertEqual(len(at.sidebar.radio), 0, "Radio selector should not be present")
        self.assertEqual(len(at.sidebar.file_uploader), 0, "Separate file_uploader boxes should not be present")

        # Before any upload, info message is displayed
        self.assertGreater(len(at.info), 0)
        self.assertIn("Upload a repository", at.info[0].value)

    def test_app_unified_uploader_zip_automatic_detection(self):
        """Verify unified uploader automatically detects and indexes a dropped/selected ZIP archive."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        zip_buf = self._create_sample_repo_zip()

        at = AppTest.from_file(str(app_path))
        at.run()

        # Deliver ZIP upload to unified uploader
        at.session_state["codebase_uploader"] = [MockUploadedFile("my-repo.zip", zip_buf.getvalue())]
        at.run()

        self.assertFalse(at.exception, f"App crashed on ZIP upload: {at.exception}")
        self.assertIsNotNone(at.session_state.get("retriever"))
        summary = at.session_state.get("repo_summary")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.repo_name, "my-repo")
        self.assertEqual(summary.total_files, 4)

    def test_app_unified_uploader_folder_automatic_detection(self):
        """Verify unified uploader automatically detects and indexes a dropped/selected folder."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"

        at = AppTest.from_file(str(app_path))
        at.run()

        # Deliver folder upload (relative paths containing '/')
        at.session_state["codebase_uploader"] = [
            MockUploadedFile("custom_proj/src/main.py", b"def main(): pass\n"),
            MockUploadedFile("custom_proj/README.md", b"# Custom Proj\n"),
        ]
        at.run()

        self.assertIsNotNone(at.session_state.get("retriever"))
        summary = at.session_state.get("repo_summary")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.repo_name, "custom_proj")
        self.assertEqual(summary.total_files, 2)

    def test_app_unified_uploader_individual_files_automatic_detection(self):
        """Verify unified uploader automatically detects and indexes flat individual files."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"

        at = AppTest.from_file(str(app_path))
        at.run()

        # Deliver flat individual files (no directory slashes)
        at.session_state["codebase_uploader"] = [
            MockUploadedFile("calc.py", b"def add(a, b): return a + b\n"),
            MockUploadedFile("test_calc.py", b"def test(): assert True\n"),
        ]
        at.run()

        self.assertIsNotNone(at.session_state.get("retriever"))
        summary = at.session_state.get("repo_summary")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.repo_name, "Uploaded files")
        self.assertEqual(summary.total_files, 2)

    def test_app_unified_uploader_ambiguous_mixed_upload_rejected(self):
        """Verify mixing a ZIP archive with other files displays a clear error and rejects indexing."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        zip_buf = self._create_sample_repo_zip()

        at = AppTest.from_file(str(app_path))
        at.run()

        # Deliver mixed ZIP + source file
        at.session_state["codebase_uploader"] = [
            MockUploadedFile("my-repo.zip", zip_buf.getvalue()),
            MockUploadedFile("extra.py", b"print('extra')\n"),
        ]
        at.run()

        self.assertGreater(len(at.sidebar.error), 0)
        self.assertIn("Ambiguous upload", at.sidebar.error[0].value)
        self.assertIsNone(at.session_state.get("retriever"))
        self.assertIsNone(at.session_state.get("repo_summary"))

    def test_app_unified_uploader_switching_formats_replaces_active_index(self):
        """Verify seamlessly switching from ZIP to folder to files updates the index automatically."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        zip_buf = self._create_sample_repo_zip()

        at = AppTest.from_file(str(app_path))
        at.run()

        # 1. Start with ZIP
        at.session_state["codebase_uploader"] = [MockUploadedFile("my-repo.zip", zip_buf.getvalue())]
        at.run()
        self.assertEqual(at.session_state["repo_summary"].repo_name, "my-repo")

        # 2. Switch directly to folder
        at.session_state["codebase_uploader"] = [
            MockUploadedFile("new_folder/src/app.py", b"def app(): pass\n")
        ]
        at.run()
        self.assertEqual(at.session_state["repo_summary"].repo_name, "new_folder")

        # 3. Switch directly to individual files
        at.session_state["codebase_uploader"] = [
            MockUploadedFile("script.py", b"def script(): pass\n")
        ]
        at.run()
        self.assertEqual(at.session_state["repo_summary"].repo_name, "Uploaded files")

    def test_app_failed_upload_clears_previous_index(self):
        """Verify replacing a valid repository with an invalid one does not retain stale results."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        zip_good = self._create_sample_repo_zip()

        zip_bad = io.BytesIO()
        with zipfile.ZipFile(zip_bad, "w") as zf:
            zf.writestr("binary.exe", b"\x7fELF")

        at = AppTest.from_file(str(app_path))
        at.run()

        # 1. Upload good ZIP
        at.session_state["codebase_uploader"] = [MockUploadedFile("good.zip", zip_good.getvalue())]
        at.run()
        self.assertIsNotNone(at.session_state.get("retriever"))

        # 2. Directly replace with bad ZIP
        at.session_state["codebase_uploader"] = [MockUploadedFile("bad.zip", zip_bad.getvalue())]
        at.run()

        self.assertGreater(len(at.sidebar.error), 0)
        # Previous index must be cleared
        self.assertIsNone(at.session_state.get("retriever"))
        self.assertIsNone(at.session_state.get("repo_summary"))

    def test_app_clearing_all_uploaders_clears_state_and_shows_prompt(self):
        """Verify clearing the uploader removes index and shows getting started message."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        zip_buf = self._create_sample_repo_zip()

        at = AppTest.from_file(str(app_path))
        at.run()

        # Upload then clear
        at.session_state["codebase_uploader"] = [MockUploadedFile("repo.zip", zip_buf.getvalue())]
        at.run()
        self.assertIsNotNone(at.session_state.get("retriever"))

        at.session_state["codebase_uploader"] = None
        at.run()
        self.assertIsNone(at.session_state.get("retriever"))
        self.assertIsNone(at.session_state.get("repo_summary"))
        self.assertIn("Upload a repository", at.info[0].value)


class TestDeriveFolderRepoName(unittest.TestCase):
    """Tests for safe folder display-name extraction from uploaded files."""

    def test_empty_list_returns_default(self):
        self.assertEqual(_derive_folder_repo_name([]), "uploaded_folder")

    def test_standard_nested_path(self):
        files = [MockUploadedFile("my_project/src/main.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "my_project")

    def test_leading_slash_normalized(self):
        files = [MockUploadedFile("/my_project/src/main.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "my_project")

    def test_leading_dot_slash_normalized(self):
        files = [MockUploadedFile("./my_project/src/main.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "my_project")

    def test_windows_backslashes_normalized(self):
        files = [MockUploadedFile("my_project\\src\\main.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "my_project")

    def test_empty_or_malformed_entries_skipped(self):
        files = [
            MockUploadedFile("", b""),
            MockUploadedFile("   ", b""),
            MockUploadedFile("real_repo/app.py", b""),
        ]
        self.assertEqual(_derive_folder_repo_name(files), "real_repo")

    def test_non_string_name_handled_safely(self):
        files = [MockUploadedFile(None, b"")]
        self.assertEqual(_derive_folder_repo_name(files), "uploaded_folder")

    def test_flat_files_without_directory_component_returns_default(self):
        files = [MockUploadedFile("main.py", b""), MockUploadedFile("utils.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "uploaded_folder")

    def test_path_traversal_component_returns_default(self):
        files = [MockUploadedFile("../evil.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "uploaded_folder")

    def test_trailing_slash_directory_entry(self):
        files = [MockUploadedFile("project_dir/", b"")]
        self.assertEqual(_derive_folder_repo_name(files), "project_dir")

    def test_delivery_order_independence(self):
        files_a = [MockUploadedFile("proj/b.py", b""), MockUploadedFile("proj/a.py", b"")]
        files_b = [MockUploadedFile("proj/a.py", b""), MockUploadedFile("proj/b.py", b"")]
        self.assertEqual(_derive_folder_repo_name(files_a), "proj")
        self.assertEqual(_derive_folder_repo_name(files_b), "proj")


class TestProcessRepositoryFiles(unittest.TestCase):
    """Tests for process_repository_files() and compute_files_upload_key()."""

    # ------------------------------------------------------------------ helpers

    def _make(self, name: str, content: bytes) -> MockUploadedFile:
        return MockUploadedFile(name, content)

    # ------------------------------------------------------------------ success

    def test_multiple_source_files_are_indexed(self):
        """Multiple Python + JS files are staged, filtered, chunked, and indexed."""
        files = [
            self._make("auth.py", b"def login(user, pwd):\n    return verify(user, pwd)\n"),
            self._make("utils.py", b"def helper():\n    return 42\n"),
            self._make("client.js", b"function init() { console.log('ready'); }\n"),
        ]
        retriever, summary = process_repository_files(files, repo_name="Uploaded files")

        self.assertIsInstance(retriever, TFIDFRetriever)
        self.assertIsInstance(summary, RepoSummary)
        self.assertEqual(summary.repo_name, "Uploaded files")
        self.assertEqual(summary.total_files, 3)
        self.assertGreaterEqual(summary.total_chunks, 3)
        self.assertIn("python", summary.language_breakdown)
        self.assertIn("javascript", summary.language_breakdown)

    def test_nested_relative_paths_are_preserved(self):
        """Files with nested paths (as returned by directory upload) are indexed correctly."""
        files = [
            self._make("my_app/src/main.py", b"def run():\n    pass\n"),
            self._make("my_app/tests/test_main.py", b"def test_run():\n    assert True\n"),
            self._make("my_app/README.md", b"# My App\nDocs here.\n"),
        ]
        retriever, summary = process_repository_files(files, repo_name="Uploaded files")

        self.assertEqual(summary.total_files, 3)
        self.assertIn("my_app/src/main.py", summary.file_list)
        self.assertIn("my_app/tests/test_main.py", summary.file_list)
        self.assertIn("my_app/README.md", summary.file_list)
        self.assertTrue(retriever.is_indexed)

    def test_unsupported_files_are_ignored(self):
        """Unsupported file types (.exe, .png) are silently filtered out."""
        files = [
            self._make("main.py", b"def main():\n    pass\n"),
            self._make("binary.exe", b"\x7fELF\x00\x01"),
            self._make("image.png", b"\x89PNG\r\n"),
        ]
        retriever, summary = process_repository_files(files, repo_name="Uploaded files")

        self.assertEqual(summary.total_files, 1)
        self.assertIn("main.py", summary.file_list)

    def test_search_works_after_individual_file_upload(self):
        """Retriever built from individual files returns relevant results."""
        files = [
            self._make("db.py", b"def connect_database(host, port):\n    return create_conn(host, port)\n"),
            self._make("auth.py", b"def authenticate(user, password):\n    return db.check(user, password)\n"),
        ]
        retriever, _ = process_repository_files(files, repo_name="Uploaded files")

        results = retriever.search("database connection", top_k=1)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0].score, 0.0)
        self.assertEqual(results[0].chunk.file_path, "db.py")

    # ------------------------------------------------------------------ errors

    def test_empty_file_list_raises_value_error(self):
        """Empty list raises ValueError with 'empty' in the message."""
        with self.assertRaises(ValueError) as ctx:
            process_repository_files([], repo_name="Uploaded files")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_all_unsupported_files_raises_value_error(self):
        """A list with only unsupported files raises ValueError."""
        files = [
            self._make("binary.exe", b"\x7fELF\x00"),
            self._make("image.png", b"\x89PNG\r\n"),
        ]
        with self.assertRaises(ValueError) as ctx:
            process_repository_files(files, repo_name="Uploaded files")
        self.assertIn("No supported source code files", str(ctx.exception))

    def test_duplicate_filenames_raises_security_error(self):
        """Two files with the same name raise SecurityError (not silent overwrite)."""
        files = [
            self._make("main.py", b"def first(): pass\n"),
            self._make("main.py", b"def second(): pass\n"),
        ]
        with self.assertRaises(SecurityError):
            process_repository_files(files, repo_name="Uploaded files")

    def test_path_traversal_in_individual_file_raises_security_error(self):
        """Path traversal in individual-file names is rejected."""
        files = [self._make("../evil.py", b"evil")]
        with self.assertRaises(SecurityError):
            process_repository_files(files, repo_name="Uploaded files")

    # ------------------------------------------------------------------ naming

    def test_repo_name_passed_through_to_summary(self):
        """summary.repo_name reflects whatever repo_name argument is provided."""
        files = [self._make("app.py", b"x = 1\n")]
        _, summary = process_repository_files(files, repo_name="Uploaded files")
        self.assertEqual(summary.repo_name, "Uploaded files")

    # ------------------------------------------------------------------ key tests

    def test_compute_files_upload_key_prefix(self):
        """Key is prefixed with 'files_'."""
        key = compute_files_upload_key([("a.py", b"content")])
        self.assertTrue(key.startswith("files_"))

    def test_compute_files_upload_key_order_independent(self):
        """Same files in different order produce the same key."""
        item1 = ("a.py", b"hello")
        item2 = ("b.py", b"world")
        k1 = compute_files_upload_key([item1, item2])
        k2 = compute_files_upload_key([item2, item1])
        self.assertEqual(k1, k2)

    def test_compute_files_upload_key_content_sensitive(self):
        """Changed content produces a different key."""
        k1 = compute_files_upload_key([("a.py", b"hello")])
        k2 = compute_files_upload_key([("a.py", b"hello CHANGED")])
        self.assertNotEqual(k1, k2)

    def test_compute_files_upload_key_distinct_from_folder_key(self):
        """files_ key is always different from a folder_ key for the same content."""
        items = [("a.py", b"content")]
        files_key = compute_files_upload_key(items)
        folder_key = compute_folder_upload_key("repo", items)
        self.assertNotEqual(files_key, folder_key)
        self.assertTrue(files_key.startswith("files_"))
        self.assertTrue(folder_key.startswith("folder_"))

    def test_compute_files_upload_key_distinct_from_zip_key(self):
        """files_ key is always different from a zip_ key."""
        files_key = compute_files_upload_key([("a.py", b"content")])
        zip_key = compute_upload_key("a.zip", b"content")
        self.assertNotEqual(files_key, zip_key)

    # ------------------------------------------------------------------ stale index

    def test_switching_from_folder_to_files_produces_different_key(self):
        """Upload keys for folder vs individual-files modes are always distinct,
        preventing stale index reuse when the user switches upload types."""
        items = [("src/main.py", b"def foo(): pass\n")]
        folder_key = compute_folder_upload_key("src", items)
        files_key = compute_files_upload_key(items)
        self.assertNotEqual(folder_key, files_key)

    def test_switching_from_zip_to_files_produces_different_key(self):
        """ZIP key and files key are distinct for the same raw content."""
        content = b"def foo(): pass\n"
        zip_key = compute_upload_key("repo.zip", content)
        files_key = compute_files_upload_key([("main.py", content)])
        self.assertNotEqual(zip_key, files_key)

    def test_different_upload_types_have_distinct_key_prefixes(self):
        """zip_, folder_, and files_ prefixes guarantee no cross-type key collision."""
        content = b"same content"
        z = compute_upload_key("r.zip", content)
        f = compute_folder_upload_key("r", [("a.py", content)])
        i = compute_files_upload_key([("a.py", content)])
        prefixes = {z.split("_")[0], f.split("_")[0], i.split("_")[0]}
        self.assertEqual(prefixes, {"zip", "folder", "files"})
        # All three must be different keys
        self.assertNotEqual(z, f)
        self.assertNotEqual(f, i)
        self.assertNotEqual(z, i)


class TestUnifiedUploadClassification(unittest.TestCase):
    """Tests for automatic upload detection and parsing in the unified uploader pipeline."""

    def test_parse_unified_upload_base64_dict_payload(self):
        """Verify custom component base64 dict payload is parsed into file objects."""
        import base64
        b64_content = base64.b64encode(b"def foo(): pass\n").decode("ascii")
        raw = [{"name": "app/main.py", "content": b64_content, "size": 16}]
        parsed = parse_unified_upload(raw)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].name, "app/main.py")
        self.assertEqual(parsed[0].getvalue(), b"def foo(): pass\n")

    def test_parse_unified_upload_empty_inputs(self):
        """Verify None, empty list, and empty dict are handled cleanly."""
        self.assertEqual(parse_unified_upload(None), [])
        self.assertEqual(parse_unified_upload([]), [])
        self.assertEqual(parse_unified_upload({}), [])

    def test_classify_single_zip(self):
        """Verify single ZIP archive is classified as ZIP and derived name is stem."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("my_app/main.py", "def main(): pass\n")
        items = [MockUploadedFile("my_app.zip", buf.getvalue())]

        retriever, summary, key = classify_and_process_upload(items)
        self.assertEqual(summary.repo_name, "my_app")
        self.assertTrue(key.startswith("zip_my_app.zip_"))
        self.assertTrue(retriever.is_indexed)

    def test_classify_folder_with_relative_paths(self):
        """Verify files with relative paths are classified as folder upload."""
        items = [
            MockUploadedFile("project/src/a.py", b"def a(): pass\n"),
            MockUploadedFile("project/src/b.py", b"def b(): pass\n"),
        ]
        retriever, summary, key = classify_and_process_upload(items)
        self.assertEqual(summary.repo_name, "project")
        self.assertTrue(key.startswith("folder_"))
        self.assertEqual(summary.total_files, 2)

    def test_classify_flat_individual_files(self):
        """Verify flat files without directory structure are classified as individual files."""
        items = [
            MockUploadedFile("main.py", b"def a(): pass\n"),
            MockUploadedFile("utils.py", b"def b(): pass\n"),
        ]
        retriever, summary, key = classify_and_process_upload(items)
        self.assertEqual(summary.repo_name, "Uploaded files")
        self.assertTrue(key.startswith("files_"))
        self.assertEqual(summary.total_files, 2)

    def test_classify_ambiguous_mixed_zip_and_files_raises_value_error(self):
        """Verify mixing a ZIP file with other files raises an unambiguous ValueError."""
        items = [
            MockUploadedFile("repo.zip", b"dummy zip"),
            MockUploadedFile("main.py", b"print(1)"),
        ]
        with self.assertRaises(ValueError) as ctx:
            classify_and_process_upload(items)
        self.assertIn("ambiguous upload", str(ctx.exception).lower())

    def test_classify_multiple_zips_raises_value_error(self):
        """Verify uploading multiple ZIP files raises an ambiguous ValueError."""
        items = [
            MockUploadedFile("repo1.zip", b"zip1"),
            MockUploadedFile("repo2.zip", b"zip2"),
        ]
        with self.assertRaises(ValueError) as ctx:
            classify_and_process_upload(items)
        self.assertIn("ambiguous upload", str(ctx.exception).lower())

    def test_classify_empty_upload_raises_value_error(self):
        """Verify classifying empty upload raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            classify_and_process_upload([])
        self.assertIn("empty", str(ctx.exception).lower())

    def test_compute_unified_upload_key_distinct_prefixes(self):
        """Verify compute_unified_upload_key generates correct prefix for each upload type."""
        zip_items = [MockUploadedFile("r.zip", b"content")]
        folder_items = [MockUploadedFile("dir/a.py", b"content")]
        file_items = [MockUploadedFile("a.py", b"content")]

        k_zip = compute_unified_upload_key(zip_items)
        k_folder = compute_unified_upload_key(folder_items)
        k_files = compute_unified_upload_key(file_items)

        self.assertTrue(k_zip.startswith("zip_r.zip_"))
        self.assertTrue(k_folder.startswith("folder_"))
        self.assertTrue(k_files.startswith("files_"))


if __name__ == "__main__":
    unittest.main()
