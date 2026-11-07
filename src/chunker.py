"""Source code chunking module with line-accurate metadata preservation."""

from pathlib import Path
from typing import List

from src.filter import SourceFile
from src.models import CodeChunk


class ChunkingError(Exception):
    """Raised when reading or chunking a source file fails."""
    pass


def validate_chunk_params(chunk_size: int, overlap: int) -> None:
    """Validates chunk_size and overlap parameters.

    Args:
        chunk_size: Number of lines per chunk.
        overlap: Number of overlapping lines between consecutive chunks.

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def chunk_text(
    content: str,
    file_path: str,
    language: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> List[CodeChunk]:
    """Splits raw text content into overlapping CodeChunk objects.

    Args:
        content: The text content to chunk.
        file_path: Relative path of the file for provenance metadata.
        language: Programming language identifier.
        chunk_size: Target line count per chunk (default 50).
        overlap: Overlapping line count between consecutive chunks (default 10).

    Returns:
        List of CodeChunk instances. Empty if content has no lines.

    Raises:
        ValueError: If chunk_size or overlap parameters are invalid.
    """
    validate_chunk_params(chunk_size, overlap)

    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return []

    step = chunk_size - overlap
    chunks: List[CodeChunk] = []
    start_idx = 0

    while start_idx < total_lines:
        end_idx = min(start_idx + chunk_size, total_lines)
        chunk_lines = lines[start_idx:end_idx]

        start_line = start_idx + 1
        end_line = end_idx
        chunk_content = "\n".join(chunk_lines)
        chunk_id = f"{file_path}:{start_line}-{end_line}"

        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                content=chunk_content,
                language=language,
            )
        )

        # When the current chunk reaches the end of the file, all lines are covered
        if end_idx >= total_lines:
            break

        start_idx += step

    return chunks


def chunk_file(
    source_file: SourceFile,
    chunk_size: int = 50,
    overlap: int = 10,
) -> List[CodeChunk]:
    """Reads a source file from disk and splits it into overlapping CodeChunks.

    Args:
        source_file: A SourceFile instance containing file metadata and path.
        chunk_size: Target line count per chunk (default 50).
        overlap: Overlapping line count between consecutive chunks (default 10).

    Returns:
        List of CodeChunk instances.

    Raises:
        ChunkingError: If the file cannot be read or decoded due to OS or encoding errors.
        ValueError: If chunk_size or overlap parameters are invalid.
    """
    validate_chunk_params(chunk_size, overlap)

    path = Path(source_file.full_path)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        raise ChunkingError(
            f"Failed to decode source file '{source_file.full_path}' as UTF-8: {err}"
        ) from err
    except OSError as err:
        raise ChunkingError(
            f"Failed to read source file '{source_file.full_path}': {err}"
        ) from err

    return chunk_text(
        content=content,
        file_path=source_file.relative_path,
        language=source_file.language,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def chunk_repository(
    source_files: List[SourceFile],
    chunk_size: int = 50,
    overlap: int = 10,
) -> List[CodeChunk]:
    """Chunks an entire collection of SourceFiles discovered from a repository.

    Args:
        source_files: Collection of SourceFile instances.
        chunk_size: Lines per chunk.
        overlap: Overlapping lines between consecutive chunks.

    Returns:
        Combined list of all CodeChunk instances across all files.
    """
    validate_chunk_params(chunk_size, overlap)

    all_chunks: List[CodeChunk] = []
    for sf in source_files:
        all_chunks.extend(chunk_file(sf, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
