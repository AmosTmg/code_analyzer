"""TF-IDF retrieval engine for context-aware code search."""

import re
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import CodeChunk, SearchResult


class NotIndexedError(RuntimeError):
    """Raised when a search query is executed before indexing code chunks."""
    pass


def _preprocess_code(text: str) -> str:
    """Preprocesses code content by expanding camelCase/PascalCase identifiers and lowercasing.

    Examples:
        'getUserById' -> 'get user by id'
        'authenticate_user' -> 'authenticate_user' (split by tokenizer)
    """
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return expanded.lower()


class TFIDFRetriever:
    """Retrieval engine that builds a TF-IDF index over CodeChunks and ranks queries by cosine similarity.

    Attributes:
        _chunks: List of indexed CodeChunk instances.
        _vectorizer: Fitted TfidfVectorizer instance, or None if unindexed or empty.
        _matrix: Sparse TF-IDF matrix corresponding to _chunks, or None.
        _is_indexed: Boolean tracking whether index() has been executed.
    """

    def __init__(self, chunks: Optional[List[CodeChunk]] = None) -> None:
        """Initializes the retriever, optionally indexing the provided chunks.

        Args:
            chunks: Optional list of CodeChunk objects to index immediately.
        """
        self._chunks: List[CodeChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._is_indexed: bool = False

        if chunks is not None:
            self.index(chunks)

    @property
    def is_indexed(self) -> bool:
        """Returns True if the retriever has indexed a collection (even if empty)."""
        return self._is_indexed

    @property
    def total_chunks(self) -> int:
        """Returns the number of indexed code chunks."""
        return len(self._chunks)

    @property
    def chunks(self) -> List[CodeChunk]:
        """Returns a shallow copy of the indexed code chunks."""
        return list(self._chunks)

    def index(self, chunks: List[CodeChunk]) -> None:
        """Builds the TF-IDF matrix from the provided collection of CodeChunk objects.

        Preserves the original chunks alongside the matrix for metadata retrieval.

        Args:
            chunks: Collection of CodeChunk objects to index.
        """
        self._chunks = list(chunks)
        self._is_indexed = True

        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            return

        corpus = [chunk.content for chunk in self._chunks]

        vectorizer = TfidfVectorizer(
            preprocessor=_preprocess_code,
            token_pattern=r"[a-zA-Z0-9]+",
        )

        try:
            self._matrix = vectorizer.fit_transform(corpus)
            self._vectorizer = vectorizer
        except ValueError:
            # Raised by scikit-learn if all documents contain only non-token characters or stop words
            self._vectorizer = None
            self._matrix = None

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Searches the indexed chunks for snippets most relevant to the query.

        Behavior:
          - Queries are transformed using the fitted TF-IDF vocabulary.
          - Chunks with zero cosine similarity (score <= 0.0) are excluded.
          - Results are ranked in descending order of similarity score.
          - Tied scores are broken deterministically by chunk_id.
          - Empty queries or queries with no matching terms return an empty list.

        Args:
            query: Natural language or code search query string.
            top_k: Maximum number of results to return (default 5, must be > 0).

        Returns:
            List of SearchResult objects ordered by descending cosine similarity.

        Raises:
            NotIndexedError: If search is attempted before index() has been called.
            ValueError: If top_k is not a positive integer.
        """
        if not self._is_indexed:
            raise NotIndexedError(
                "Retriever has not been indexed yet. Call index() before searching."
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # Empty or whitespace-only queries yield no results
        if not query or not query.strip():
            return []

        # If index has no valid matrix or chunks, return empty results
        if self._vectorizer is None or self._matrix is None or not self._chunks:
            return []

        # Transform query into the existing TF-IDF feature space
        query_vec = self._vectorizer.transform([query])

        # Compute cosine similarity between query and all indexed chunks
        similarities = cosine_similarity(query_vec, self._matrix).flatten()

        # Collect candidates with strictly positive similarity (filtering zero-similarity results)
        scored_candidates = []
        for idx, score in enumerate(similarities):
            val = float(score)
            if val > 0.0:
                chunk = self._chunks[idx]
                scored_candidates.append((val, chunk.chunk_id, chunk))

        # Deterministic sorting: descending score, then ascending chunk_id for tie-breaking
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        # Return top_k SearchResults
        return [
            SearchResult(chunk=chunk, score=score)
            for score, _, chunk in scored_candidates[:top_k]
        ]
