"""Unit tests for the source code chunking module."""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.chunker import (
    ChunkingError,
    chunk_file,
    chunk_repository,
    chunk_text,
    validate_chunk_params,
)
from src.filter import SourceFile


class TestChunkerModule(unittest.TestCase):
    """Test suite for line-based code chunking and metadata preservation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_source_file(
        self,
        rel_path: str,
        lines: list[str],
        language: str = "python",
        encoding: str = "utf-8",
    ) -> SourceFile:
        """Helper to create a test file and return a valid SourceFile object."""
        full_path = Path(self.temp_dir) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines)
        full_path.write_text(content, encoding=encoding)
        return SourceFile(
            relative_path=rel_path,
            full_path=full_path,
            language=language,
            size_bytes=full_path.stat().st_size,
        )

    def test_normal_chunk_creation(self):
        """Verify standard chunking on a file matching chunk_size."""
        lines = [f"line_{i}" for i in range(1, 51)]
        sf = self._create_source_file("app/main.py", lines, language="python")

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_id, "app/main.py:1-50")
        self.assertEqual(chunk.file_path, "app/main.py")
        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.end_line, 50)
        self.assertEqual(chunk.language, "python")
        self.assertEqual(chunk.content.splitlines(), lines)

    def test_correct_overlap_between_consecutive_chunks(self):
        """Verify that consecutive chunks share exactly `overlap` lines."""
        # 60 lines, chunk_size=30, overlap=10 -> step=20
        # Chunk 1: lines 1..30
        # Chunk 2: lines 21..50 (overlap with chunk 1: lines 21..30)
        # Chunk 3: lines 41..60 (overlap with chunk 2: lines 41..50)
        lines = [f"const x_{i} = {i};" for i in range(1, 61)]
        sf = self._create_source_file("src/index.ts", lines, language="typescript")

        chunks = chunk_file(sf, chunk_size=30, overlap=10)

        self.assertEqual(len(chunks), 3)

        # Chunk 1
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 30)

        # Chunk 2
        self.assertEqual(chunks[1].start_line, 21)
        self.assertEqual(chunks[1].end_line, 50)

        # Chunk 3
        self.assertEqual(chunks[2].start_line, 41)
        self.assertEqual(chunks[2].end_line, 60)

        # Verify overlapping line content between Chunk 1 and Chunk 2
        chunk1_lines = chunks[0].content.splitlines()
        chunk2_lines = chunks[1].content.splitlines()
        self.assertEqual(chunk1_lines[20:], chunk2_lines[:10])
        self.assertEqual(chunk1_lines[20:], lines[20:30])

        # Verify overlapping line content between Chunk 2 and Chunk 3
        chunk3_lines = chunks[2].content.splitlines()
        self.assertEqual(chunk2_lines[20:], chunk3_lines[:10])
        self.assertEqual(chunk2_lines[20:], lines[40:50])

    def test_start_and_end_line_values_are_one_based(self):
        """Verify 1-based indexing for start_line and end_line across all chunks."""
        lines = [f"print({i})" for i in range(1, 86)]  # 85 lines
        sf = self._create_source_file("math_ops.py", lines)

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        # Chunk 1: 1..50, Chunk 2: 41..85
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 50)
        self.assertEqual(chunks[1].start_line, 41)
        self.assertEqual(chunks[1].end_line, 85)

    def test_file_path_and_language_metadata_preserved(self):
        """Verify that relative path and language are accurately attached to every chunk."""
        lines = ["<h1>Hello</h1>", "<p>World</p>"]
        sf = self._create_source_file("views/home.html", lines, language="html")

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].file_path, "views/home.html")
        self.assertEqual(chunks[0].language, "html")

    def test_empty_file_returns_no_chunks(self):
        """Verify that an empty file produces 0 chunks without error."""
        sf = self._create_source_file("empty.py", [])

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(chunks, [])

    def test_small_file_fewer_than_chunk_size(self):
        """Verify that a file with fewer lines than chunk_size yields exactly 1 chunk."""
        lines = ["line 1", "line 2", "line 3"]
        sf = self._create_source_file("small.py", lines)

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 3)
        self.assertEqual(chunks[0].content, "line 1\nline 2\nline 3")

    def test_exact_multiple_of_chunk_size(self):
        """Verify file whose line count is an exact multiple of chunk_size."""
        lines = [f"item_{i}" for i in range(1, 101)]  # 100 lines
        sf = self._create_source_file("multiple.py", lines)

        # With overlap=10, step=40:
        # Chunk 1: lines 1..50
        # Chunk 2: lines 41..90
        # Chunk 3: lines 81..100 (final partial chunk of 20 lines)
        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 3)
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (1, 50))
        self.assertEqual((chunks[1].start_line, chunks[1].end_line), (41, 90))
        self.assertEqual((chunks[2].start_line, chunks[2].end_line), (81, 100))

        # With overlap=0, step=50: exactly 2 full chunks
        chunks_no_overlap = chunk_file(sf, chunk_size=50, overlap=0)
        self.assertEqual(len(chunks_no_overlap), 2)
        self.assertEqual((chunks_no_overlap[0].start_line, chunks_no_overlap[0].end_line), (1, 50))
        self.assertEqual((chunks_no_overlap[1].start_line, chunks_no_overlap[1].end_line), (51, 100))

    def test_final_partial_chunk(self):
        """Verify that trailing lines smaller than chunk_size form a correct partial chunk."""
        lines = [f"row_{i}" for i in range(1, 66)]  # 65 lines
        sf = self._create_source_file("partial.py", lines)

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 50)
        self.assertEqual(chunks[1].start_line, 41)
        self.assertEqual(chunks[1].end_line, 65)
        # Verify all 65 lines are covered without dropping any
        self.assertEqual(chunks[1].content.splitlines(), lines[40:65])

    def test_invalid_parameters_raise_value_error(self):
        """Verify that invalid chunk_size or overlap arguments raise ValueError."""
        with self.assertRaises(ValueError):
            validate_chunk_params(chunk_size=0, overlap=0)

        with self.assertRaises(ValueError):
            validate_chunk_params(chunk_size=-10, overlap=5)

        with self.assertRaises(ValueError):
            validate_chunk_params(chunk_size=50, overlap=-1)

        with self.assertRaises(ValueError):
            validate_chunk_params(chunk_size=50, overlap=50)

        with self.assertRaises(ValueError):
            validate_chunk_params(chunk_size=50, overlap=60)

    def test_deterministic_and_unique_chunk_ids(self):
        """Verify chunk IDs are unique within a file and deterministic across multiple runs."""
        lines = [f"line_{i}" for i in range(1, 120)]
        sf = self._create_source_file("service.py", lines)

        run1 = chunk_file(sf, chunk_size=50, overlap=10)
        run2 = chunk_file(sf, chunk_size=50, overlap=10)

        ids1 = [c.chunk_id for c in run1]
        ids2 = [c.chunk_id for c in run2]

        # Determinism
        self.assertEqual(ids1, ids2)

        # Uniqueness
        self.assertEqual(len(ids1), len(set(ids1)))

        # ID structure matches file_path:start-end
        for chunk in run1:
            expected_id = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
            self.assertEqual(chunk.chunk_id, expected_id)

    def test_unicode_source_code_preservation(self):
        """Verify Unicode content (emojis, CJK, accents) is read and chunked accurately."""
        lines = [
            "# 🚀 High-performance database client",
            "def authenticate_ユーザー():",
            "    '''München office credentials 🔐'''",
            "    return '你好，世界'",
        ]
        sf = self._create_source_file("unicode_test.py", lines)

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content.splitlines(), lines)
        self.assertIn("🚀", chunks[0].content)
        self.assertIn("你好，世界", chunks[0].content)
        self.assertIn("München", chunks[0].content)

    def test_file_reading_error_raises_chunking_error(self):
        """Verify that unreadable or missing files raise ChunkingError."""
        non_existent_path = Path(self.temp_dir) / "missing_file.py"
        sf = SourceFile(
            relative_path="missing_file.py",
            full_path=non_existent_path,
            language="python",
            size_bytes=0,
        )

        with self.assertRaises(ChunkingError):
            chunk_file(sf)

    def test_invalid_utf8_encoding_raises_chunking_error(self):
        """Verify that files with invalid UTF-8 bytes raise ChunkingError."""
        invalid_file = Path(self.temp_dir) / "corrupt.py"
        # 0xFF is an invalid UTF-8 start byte
        invalid_file.write_bytes(b"print('hello')\n\xff\xfe invalid utf8")

        sf = SourceFile(
            relative_path="corrupt.py",
            full_path=invalid_file,
            language="python",
            size_bytes=invalid_file.stat().st_size,
        )

        with self.assertRaises(ChunkingError) as ctx:
            chunk_file(sf)

        self.assertIn("Failed to decode", str(ctx.exception))
        self.assertIn("corrupt.py", str(ctx.exception))

    def test_chunk_repository_aggregates_multiple_files(self):
        """Verify chunk_repository collects chunks across multiple source files."""
        sf1 = self._create_source_file("file1.py", [f"a_{i}" for i in range(1, 31)])
        sf2 = self._create_source_file("file2.js", [f"b_{i}" for i in range(1, 61)], language="javascript")

        all_chunks = chunk_repository([sf1, sf2], chunk_size=50, overlap=10)

        # file1 has 30 lines -> 1 chunk
        # file2 has 60 lines -> 2 chunks (1-50, 41-60)
        self.assertEqual(len(all_chunks), 3)

        paths = [c.file_path for c in all_chunks]
        self.assertEqual(paths, ["file1.py", "file2.js", "file2.js"])

        langs = [c.language for c in all_chunks]
        self.assertEqual(langs, ["python", "javascript", "javascript"])

    def test_context_string_formatting(self):
        """Verify CodeChunk.to_context_string() provides expected structured representation."""
        lines = ["def add(a, b):", "    return a + b"]
        sf = self._create_source_file("math.py", lines)

        chunks = chunk_file(sf, chunk_size=50, overlap=10)

        self.assertEqual(len(chunks), 1)
        context_str = chunks[0].to_context_string()
        expected = (
            "--- File: math.py (Lines 1-2) [python] ---\n"
            "def add(a, b):\n"
            "    return a + b"
        )
        self.assertEqual(context_str, expected)


if __name__ == "__main__":
    unittest.main()
