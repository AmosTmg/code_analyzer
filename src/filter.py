"""Repository file filtering and language detection module."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


# Initial supported extensions and their canonical language mappings
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
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

# Directories to skip during scanning
IGNORED_DIRECTORIES: Set[str] = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
}

# Specific filenames to ignore (lockfiles, generated meta-files)
IGNORED_FILENAMES: Set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
}

# Suffixes for minified or bundled code that should not be indexed
IGNORED_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    ".map",
)


@dataclass(frozen=True)
class SourceFile:
    """Represents a discovered and validated source code file.

    Attributes:
        relative_path: POSIX relative path from repository root (e.g., 'src/main.py').
        full_path: Absolute resolved Path to the file on disk.
        language: Detected programming language.
        size_bytes: Size of the file in bytes.
    """

    relative_path: str
    full_path: Path
    language: str
    size_bytes: int


def get_language(file_path: Union[str, Path]) -> Optional[str]:
    """Resolves the programming language for a given file path based on its extension.

    Args:
        file_path: File name or path.

    Returns:
        The detected language name in lowercase, or None if unsupported.
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def is_ignored_directory(dir_name: str) -> bool:
    """Checks if a directory should be excluded from traversal.

    Args:
        dir_name: Directory name (basename).

    Returns:
        True if the directory is in the ignore list.
    """
    clean_name = dir_name.strip().lower()
    return clean_name in IGNORED_DIRECTORIES


def is_ignored_file(file_name: str) -> bool:
    """Checks if a specific file should be ignored.

    Args:
        file_name: File name (basename).

    Returns:
        True if the file is hidden, a lockfile, or a bundled/minified artifact.
    """
    clean_name = file_name.strip()
    lower_name = clean_name.lower()

    # Ignore hidden files (e.g. .DS_Store, .gitignore)
    if clean_name.startswith("."):
        return True

    # Ignore exact lockfiles
    if lower_name in IGNORED_FILENAMES:
        return True

    # Ignore minified and bundled artifacts
    if any(lower_name.endswith(suffix) for suffix in IGNORED_SUFFIXES):
        return True

    return False


def is_binary_file(file_path: Path, sample_size: int = 8192) -> bool:
    """Detects whether a file contains binary content rather than plain text.

    Args:
        file_path: Absolute or relative Path to the file.
        sample_size: Number of bytes to inspect at the beginning of the file.

    Returns:
        True if binary content or unreadable bytes are detected, False otherwise.
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
            if not chunk:
                return False

            # Null byte presence is a strong indicator of binary content
            if b"\x00" in chunk:
                return True

            # Attempt UTF-8 decoding on the sample
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                # If UTF-8 fails, check non-text byte ratio using allowed text bytes
                # Bytes allowed in text: ASCII control chars (\a,\b,\t,\n,\f,\r,\x1b) and printable/extended bytes
                allowed_text_bytes = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
                non_text = chunk.translate(None, allowed_text_bytes)
                if len(non_text) / len(chunk) > 0.30:
                    return True

        return False
    except (OSError, PermissionError):
        # Treat unreadable files as non-indexable
        return True


def filter_repository(repo_dir: Union[str, Path]) -> List[SourceFile]:
    """Recursively scans a repository directory and returns supported, valid source files.

    Performs:
      - Recursive directory scanning with pruning of ignored directories
      - Extension filtering against supported languages
      - Exclusion of lockfiles, minified files, and hidden files
      - Binary content detection
      - Deterministic sorting by relative path

    Args:
        repo_dir: Path to the root of the extracted repository.

    Returns:
        List of SourceFile objects sorted alphabetically by relative path.

    Raises:
        FileNotFoundError: If repo_dir does not exist.
        NotADirectoryError: If repo_dir is not a directory.
    """
    root_path = Path(repo_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Repository directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root_path}")

    discovered_files: List[SourceFile] = []

    for current_root, dirnames, filenames in os.walk(root_path, topdown=True):
        current_root_path = Path(current_root)

        # Prune ignored directories and symbolic links in-place
        dirnames[:] = [
            d for d in dirnames
            if not is_ignored_directory(d) and not (current_root_path / d).is_symlink()
        ]

        for filename in filenames:
            if is_ignored_file(filename):
                continue

            lang = get_language(filename)
            if lang is None:
                # Unsupported extension
                continue

            full_file_path = current_root_path / filename

            # Skip symbolic links to files
            if full_file_path.is_symlink():
                continue

            # Guard against unreadable or binary files
            if is_binary_file(full_file_path):
                continue

            try:
                size = full_file_path.stat().st_size
                rel_path = full_file_path.relative_to(root_path).as_posix()
            except (OSError, ValueError):
                continue

            discovered_files.append(
                SourceFile(
                    relative_path=rel_path,
                    full_path=full_file_path,
                    language=lang,
                    size_bytes=size,
                )
            )

    # Sort deterministically by relative path
    discovered_files.sort(key=lambda sf: sf.relative_path)
    return discovered_files
