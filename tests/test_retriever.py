"""Unit tests for the TF-IDF retrieval engine."""

import unittest

from src.models import CodeChunk, SearchResult
from src.retriever import NotIndexedError, TFIDFRetriever


class TestTFIDFRetriever(unittest.TestCase):
    """Test suite for TF-IDF indexing, cosine ranking, and metadata preservation."""

    def _create_chunk(
        self,
        chunk_id: str,
        content: str,
        file_path: str = "src/example.py",
        start_line: int = 1,
        end_line: int = 10,
        language: str = "python",
    ) -> CodeChunk:
        """Helper to create a CodeChunk with test metadata."""
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            language=language,
        )

    def test_indexing_and_properties(self):
        """Verify indexing stores chunks, marks retriever as indexed, and exposes properties."""
        chunks = [
            self._create_chunk("chunk_1", "def add(a, b): return a + b"),
            self._create_chunk("chunk_2", "def sub(a, b): return a - b"),
        ]
        retriever = TFIDFRetriever(chunks)

        self.assertTrue(retriever.is_indexed)
        self.assertEqual(retriever.total_chunks, 2)
        self.assertEqual(len(retriever.chunks), 2)
        self.assertEqual(retriever.chunks[0].chunk_id, "chunk_1")

    def test_search_before_indexing_raises_not_indexed_error(self):
        """Verify searching an unindexed retriever raises NotIndexedError."""
        retriever = TFIDFRetriever()
        self.assertFalse(retriever.is_indexed)

        with self.assertRaises(NotIndexedError):
            retriever.search("any query")

    def test_empty_chunk_collection_returns_empty_results(self):
        """Verify an indexed empty collection returns an empty list without error."""
        retriever = TFIDFRetriever([])
        self.assertTrue(retriever.is_indexed)
        self.assertEqual(retriever.total_chunks, 0)

        results = retriever.search("query")
        self.assertEqual(results, [])

    def test_empty_or_whitespace_query_returns_empty_results(self):
        """Verify blank or whitespace-only queries return no results."""
        chunks = [self._create_chunk("c1", "print('hello world')")]
        retriever = TFIDFRetriever(chunks)

        self.assertEqual(retriever.search(""), [])
        self.assertEqual(retriever.search("   "), [])
        self.assertEqual(retriever.search("\t\n"), [])

    def test_invalid_top_k_raises_value_error(self):
        """Verify non-positive, non-integer, or boolean top_k values raise ValueError."""
        chunks = [self._create_chunk("c1", "print('hello')")]
        retriever = TFIDFRetriever(chunks)

        with self.assertRaises(ValueError):
            retriever.search("hello", top_k=0)

        with self.assertRaises(ValueError):
            retriever.search("hello", top_k=-5)

        with self.assertRaises(ValueError):
            retriever.search("hello", top_k="three")  # type: ignore

        with self.assertRaises(ValueError):
            retriever.search("hello", top_k=True)

        with self.assertRaises(ValueError):
            retriever.search("hello", top_k=False)

    def test_no_match_query_returns_empty_list(self):
        """Verify that queries with zero similarity across all chunks return an empty list."""
        chunks = [
            self._create_chunk("c1", "def calculate_factorial(n): return math.factorial(n)"),
            self._create_chunk("c2", "def matrix_multiply(a, b): return np.dot(a, b)"),
        ]
        retriever = TFIDFRetriever(chunks)

        # Query terms completely absent from corpus
        results = retriever.search("docker container kubernetes pod")
        self.assertEqual(results, [])

    def test_relevance_ranking_and_scores(self):
        """Verify that most relevant chunks are ranked first with positive cosine scores."""
        chunk_auth = self._create_chunk(
            "auth:1-5",
            "def authenticate_user(username, password):\n    validate_credentials(username)",
            file_path="src/auth.py",
        )
        chunk_db = self._create_chunk(
            "db:1-5",
            "def connect_database(host, port, db_name):\n    return create_connection(host)",
            file_path="src/db.py",
        )
        chunk_ui = self._create_chunk(
            "ui:1-5",
            "def render_dashboard(user_theme):\n    return load_template('dashboard.html')",
            file_path="src/ui.py",
        )

        retriever = TFIDFRetriever([chunk_auth, chunk_db, chunk_ui])

        results = retriever.search("database connection host", top_k=3)

        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertEqual(top_result.chunk.chunk_id, "db:1-5")
        self.assertEqual(top_result.chunk.file_path, "src/db.py")
        self.assertGreater(top_result.score, 0.0)
        self.assertLessEqual(top_result.score, 1.0)

        # Check descending order
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limiting(self):
        """Verify results are limited to the requested top_k count."""
        chunks = [
            self._create_chunk(f"c_{i}", f"logger.info('event number {i} occurred')")
            for i in range(10)
        ]
        retriever = TFIDFRetriever(chunks)

        results = retriever.search("logger event", top_k=4)
        self.assertEqual(len(results), 4)

        results_all = retriever.search("logger event", top_k=20)
        self.assertEqual(len(results_all), 10)

    def test_metadata_preservation(self):
        """Verify all original CodeChunk fields and SearchResult structure are fully preserved."""
        original_chunk = CodeChunk(
            chunk_id="models.py:10-25",
            file_path="src/database/models.py",
            start_line=10,
            end_line=25,
            content="class UserProfile(BaseModel):\n    id: int\n    name: str",
            language="python",
        )
        retriever = TFIDFRetriever([original_chunk])

        results = retriever.search("UserProfile BaseModel")

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIsInstance(res, SearchResult)
        self.assertEqual(res.chunk.chunk_id, "models.py:10-25")
        self.assertEqual(res.chunk.file_path, "src/database/models.py")
        self.assertEqual(res.chunk.start_line, 10)
        self.assertEqual(res.chunk.end_line, 25)
        self.assertEqual(res.chunk.language, "python")
        self.assertEqual(res.chunk.content, original_chunk.content)
        self.assertIsInstance(res.score, float)

    def test_deterministic_tie_breaking(self):
        """Verify that tied similarity scores are broken deterministically by chunk_id."""
        # Two identical chunks in different files
        chunk_b = self._create_chunk("b_file.py:1-5", "print('status check')", file_path="b_file.py")
        chunk_a = self._create_chunk("a_file.py:1-5", "print('status check')", file_path="a_file.py")

        # Insert chunk_b first to test sorting order
        retriever = TFIDFRetriever([chunk_b, chunk_a])

        results1 = retriever.search("status check", top_k=2)
        results2 = retriever.search("status check", top_k=2)

        self.assertEqual(len(results1), 2)
        # Scores are identical
        self.assertAlmostEqual(results1[0].score, results1[1].score, places=6)

        # Lexicographical tie-break: 'a_file.py:1-5' must precede 'b_file.py:1-5'
        self.assertEqual(results1[0].chunk.chunk_id, "a_file.py:1-5")
        self.assertEqual(results1[1].chunk.chunk_id, "b_file.py:1-5")

        # Deterministic across multiple calls
        self.assertEqual(
            [r.chunk.chunk_id for r in results1],
            [r.chunk.chunk_id for r in results2],
        )

    def test_camelcase_and_identifier_expansion(self):
        """Verify natural language queries match camelCase code tokens."""
        chunk = self._create_chunk(
            "service:1-5",
            "class UserService:\n    def getUserAccountBalance(self, userId): pass",
        )
        retriever = TFIDFRetriever([chunk])

        # Query with separated words
        results = retriever.search("user account balance")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].score, 0.0)

    def test_chunks_with_only_punctuation_or_empty_text(self):
        """Verify indexing chunks that produce empty vocabulary does not crash."""
        chunks = [
            self._create_chunk("p1", ";;; ;;;"),
            self._create_chunk("p2", "--- ---"),
        ]
        retriever = TFIDFRetriever(chunks)
        self.assertTrue(retriever.is_indexed)

        # Should safely return [] rather than raising sklearn's empty vocabulary error
        results = retriever.search("anything")
        self.assertEqual(results, [])

    def test_reindexing_replaces_previous_index_and_results(self):
        """Verify that calling index() with new chunks completely replaces old index and results."""
        v1_chunks = [
            self._create_chunk("v1_1", "def authenticate_admin(): pass"),
            self._create_chunk("v1_2", "def validate_token(): pass"),
        ]
        retriever = TFIDFRetriever(v1_chunks)
        self.assertEqual(retriever.total_chunks, 2)

        res_v1 = retriever.search("authenticate")
        self.assertEqual(len(res_v1), 1)
        self.assertEqual(res_v1[0].chunk.chunk_id, "v1_1")

        # Re-index with completely new set of chunks
        v2_chunks = [
            self._create_chunk("v2_1", "def render_canvas(): pass"),
        ]
        retriever.index(v2_chunks)
        self.assertEqual(retriever.total_chunks, 1)

        # Old query returns no results now
        self.assertEqual(retriever.search("authenticate"), [])

        # New query returns new chunk
        res_v2 = retriever.search("canvas")
        self.assertEqual(len(res_v2), 1)
        self.assertEqual(res_v2[0].chunk.chunk_id, "v2_1")

    def test_small_positive_similarity_not_discarded(self):
        """Verify that legitimate small positive similarity scores (> 0.0) are not discarded."""
        # A large document with many unique terms where query matches only one term
        content = "rare_keyword " + " ".join(f"filler_{i}" for i in range(5000))
        chunk = self._create_chunk("c_rare", content)
        retriever = TFIDFRetriever([chunk])

        # Query also has many unique non-matching terms
        query = "rare_keyword " + " ".join(f"query_filler_{i}" for i in range(5000))
        results = retriever.search(query)

        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].score, 0.0)
        self.assertEqual(results[0].chunk.chunk_id, "c_rare")


if __name__ == "__main__":
    unittest.main()
