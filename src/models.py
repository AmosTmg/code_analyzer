"""Core data models representing code chunks, search results, and repository metadata."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CodeChunk:
    """Represents an extracted snippet of source code with exact provenance metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk (e.g., 'src/main.py:1-50').
        file_path: Relative path of the source file from repository root.
        start_line: Starting line number in the source file (1-based, inclusive).
        end_line: Ending line number in the source file (1-based, inclusive).
        content: The actual source code content.
        language: Programming language identifier (e.g., 'python', 'javascript').
    """

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str

    def to_context_string(self) -> str:
        """Formats the chunk into a standardized context block.
        
        Designed for modular downstream consumption (e.g., future LLM prompts).
        """
        header = f"--- File: {self.file_path} (Lines {self.start_line}-{self.end_line}) [{self.language}] ---"
        return f"{header}\n{self.content}"


@dataclass
class SearchResult:
    """Represents a retrieved code chunk paired with its relevance score.
    
    Attributes:
        chunk: The retrieved CodeChunk instance.
        score: Relevance score (e.g., cosine similarity, higher is better).
    """

    chunk: CodeChunk
    score: float


@dataclass
class RepoSummary:
    """Summary statistics and structure of an analyzed repository.
    
    Attributes:
        repo_name: Cleaned name of the repository.
        total_files: Count of supported source files processed.
        total_chunks: Count of code chunks indexed.
        file_list: List of supported relative file paths.
        language_breakdown: Mapping of programming language name to file count.
    """

    repo_name: str
    total_files: int
    total_chunks: int
    file_list: List[str] = field(default_factory=list)
    language_breakdown: Dict[str, int] = field(default_factory=dict)
