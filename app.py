"""Context-Aware Codebase Analyzer - Streamlit Application.

Provides a web interface to upload repositories as ZIP archives, directory uploads,
or individual source files, inspect repository statistics, and perform TF-IDF based
code snippet retrieval.
"""

import base64
import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Optional, Tuple, Union

import streamlit as st
import streamlit.components.v1 as components

from src.chunker import chunk_repository
from src.extractor import ExtractionError, SecurityError, safe_extract_zip
from src.filter import filter_repository
from src.models import RepoSummary
from src.retriever import TFIDFRetriever

_COMPONENT_PATH = Path(__file__).resolve().parent / "src" / "components" / "unified_uploader"
_unified_uploader_component = components.declare_component(
    "unified_uploader",
    path=str(_COMPONENT_PATH),
)


class UploadedItem:
    """Standardized representation of an uploaded file object."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def read(self, *args, **kwargs) -> bytes:
        return self._data

    def seek(self, *args, **kwargs):
        pass


def compute_upload_key(filename: str, content: bytes) -> str:
    """Computes a deterministic cache key for a ZIP file upload.

    Args:
        filename: Name of the uploaded file.
        content: Raw byte contents of the file.

    Returns:
        Deterministic identifier string prefixed with 'zip_'.
    """
    content_hash = hashlib.sha256(content).hexdigest()
    return f"zip_{filename}_{content_hash}"


def compute_folder_upload_key(
    folder_name_or_items: Union[str, list],
    file_items: Optional[list] = None,
) -> str:
    """Computes a deterministic cache key for a folder upload using length-prefixed serialization.

    Sorts files deterministically by normalized relative path and hashes:
      - Upload mode identifier ('mode:folder\n')
      - For each file:
        - Encoded normalized path length (8 bytes big-endian)
        - Encoded normalized path bytes
        - Content length (8 bytes big-endian)
        - Content bytes

    Display names are excluded from the hash so content identification is strictly content-based.

    Args:
        folder_name_or_items: Legacy folder_name string or collection of uploaded file items.
        file_items: Collection of uploaded file objects or (rel_path, content_bytes) tuples.

    Returns:
        Deterministic identifier string prefixed with 'folder_'.
    """
    if file_items is None:
        items = folder_name_or_items if isinstance(folder_name_or_items, (list, tuple)) else []
    else:
        items = file_items

    extracted_items = []
    for item in items:
        if isinstance(item, tuple):
            rel_path, content = item
        else:
            rel_path = getattr(item, "name", "")
            content = item.getvalue() if hasattr(item, "getvalue") else item.read()

        norm_path = os.path.normpath(rel_path.replace("\\", "/")).replace("\\", "/")
        extracted_items.append((norm_path, content))

    # Sort deterministically by normalized relative path
    extracted_items.sort(key=lambda x: x[0])

    hasher = hashlib.sha256()
    hasher.update(b"mode:folder\n")

    for norm_path, content in extracted_items:
        path_bytes = norm_path.encode("utf-8")
        hasher.update(len(path_bytes).to_bytes(8, byteorder="big"))
        hasher.update(path_bytes)
        hasher.update(len(content).to_bytes(8, byteorder="big"))
        hasher.update(content)

    return f"folder_{hasher.hexdigest()}"


def safe_write_folder_files(
    uploaded_files: list,
    destination_dir: Union[str, Path],
) -> Path:
    """Safely writes files from an untrusted folder upload to a destination directory.

    Guards against:
      - Empty folder uploads
      - Null bytes in file paths
      - Absolute paths (/etc/passwd, \\Windows)
      - Parent directory traversal (../)
      - Windows drive letters and alternate data streams (C:\\, D:, file:stream)
      - Duplicate normalized paths (e.g., 'src/main.py' vs 'src/./main.py')
      - Paths escaping destination_dir

    Args:
        uploaded_files: List of uploaded file objects with .name and .getvalue().
        destination_dir: Target directory path to write the folder files into.

    Returns:
        Resolved Path to destination_dir.

    Raises:
        ValueError: If uploaded_files is empty.
        SecurityError: If any file path attempts path traversal, absolute path,
                       Windows drive syntax, contains null bytes, contains duplicates,
                       or creates file/directory collisions.
    """
    if not uploaded_files:
        raise ValueError("The uploaded folder is empty.")

    dest_path = Path(destination_dir).resolve()

    seen_files: set[str] = set()
    seen_dirs: set[str] = set()
    validated_files: list[Tuple[Any, Path]] = []

    for item in uploaded_files:
        rel_name = getattr(item, "name", "")
        if not rel_name:
            raise SecurityError("Empty file path detected in folder upload.")

        # Reject null bytes in path
        if "\0" in rel_name:
            raise SecurityError(f"Null byte detected in file path: {repr(rel_name)}")

        # Reject absolute paths (POSIX and Windows)
        if os.path.isabs(rel_name) or rel_name.startswith("/") or rel_name.startswith("\\"):
            raise SecurityError(
                f"Absolute path detected in folder upload: '{rel_name}'"
            )

        # Reject Windows drive syntax and alternate data streams (colons)
        if re.match(r"^[a-zA-Z]:", rel_name) or ":" in rel_name:
            raise SecurityError(
                f"Windows drive or invalid character detected in path: '{rel_name}'"
            )

        # Convert backslashes to forward slashes for consistent cross-platform validation
        slash_path = rel_name.replace("\\", "/")

        # Inspect raw path components before normalization for parent traversal
        raw_components = slash_path.split("/")
        if any(comp == ".." for comp in raw_components):
            raise SecurityError(
                f"Path traversal ('..') detected in folder upload: '{rel_name}'"
            )

        # Normalize path separators and components
        normalized_path = os.path.normpath(slash_path).replace("\\", "/")

        if normalized_path == "." or normalized_path == "":
            raise SecurityError(f"Malformed or root path detected in folder upload: '{rel_name}'")

        target_path = (dest_path / normalized_path).resolve()

        try:
            common = os.path.commonpath([str(dest_path), str(target_path)])
        except ValueError as err:
            raise SecurityError(
                f"Cross-device or invalid path detected: '{rel_name}'"
            ) from err

        if common != str(dest_path):
            raise SecurityError(
                f"Path traversal detected in folder upload: '{rel_name}'"
            )

        # Reject duplicate normalized paths
        if normalized_path in seen_files:
            raise SecurityError(
                f"Duplicate relative path detected in folder upload: '{rel_name}' (normalized: '{normalized_path}')"
            )

        # Detect file/directory path collisions
        # Case A: An earlier file occupies a path this file requires as an ancestor directory
        parts = normalized_path.split("/")
        ancestors = ["/".join(parts[:i]) for i in range(1, len(parts))]

        for anc in ancestors:
            if anc in seen_files:
                raise SecurityError(
                    f"Path conflict: directory '{anc}' required by '{normalized_path}' conflicts with existing file '{anc}'"
                )

        # Case B: This file occupies a path that earlier files already established as a directory
        if normalized_path in seen_dirs:
            raise SecurityError(
                f"Path conflict: file '{normalized_path}' conflicts with existing directory required by other files"
            )

        seen_files.add(normalized_path)
        seen_dirs.update(ancestors)
        validated_files.append((item, target_path))

    # All files validated; safely write to disk
    dest_path.mkdir(parents=True, exist_ok=True)
    for item, target_path in validated_files:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = item.getvalue() if hasattr(item, "getvalue") else item.read()
        target_path.write_bytes(content)

    return dest_path


def process_repository_dir(
    repo_dir: Union[str, Path],
    repo_name: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> Tuple[TFIDFRetriever, RepoSummary]:
    """Common processing pipeline for an extracted or staged repository directory.

    Performs:
      1. filter_repository(repo_dir)
      2. chunk_repository(source_files, chunk_size, overlap)
      3. TFIDFRetriever(chunks)
      4. RepoSummary creation

    Args:
        repo_dir: Path to directory containing repository files.
        repo_name: Display name for the repository.
        chunk_size: Lines per chunk.
        overlap: Overlapping lines between consecutive chunks.

    Returns:
        Tuple of (TFIDFRetriever, RepoSummary).

    Raises:
        ValueError: If no supported source files or non-empty chunks exist.
    """
    source_files = filter_repository(repo_dir)
    if not source_files:
        raise ValueError(
            "No supported source code files were found in the uploaded repository."
        )

    chunks = chunk_repository(source_files, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(
            "Supported source files were found, but no non-empty code chunks could be created."
        )

    retriever = TFIDFRetriever(chunks)

    # Count language distribution
    language_breakdown = {}
    for sf in source_files:
        language_breakdown[sf.language] = language_breakdown.get(sf.language, 0) + 1

    summary = RepoSummary(
        repo_name=repo_name,
        total_files=len(source_files),
        total_chunks=len(chunks),
        file_list=[sf.relative_path for sf in source_files],
        language_breakdown=language_breakdown,
    )

    return retriever, summary


def process_repository_archive(
    zip_source: Union[str, Path, BinaryIO],
    repo_name: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> Tuple[TFIDFRetriever, RepoSummary]:
    """Safely extracts and processes a repository ZIP archive.

    Args:
        zip_source: Path to ZIP file or file-like binary stream.
        repo_name: Clean display name for the repository.
        chunk_size: Number of lines per chunk.
        overlap: Overlapping lines between consecutive chunks.

    Returns:
        Tuple containing the fitted TFIDFRetriever and a RepoSummary.

    Raises:
        ExtractionError: If the archive is empty, invalid, corrupted, or unsafe.
        ValueError: If no supported files or non-empty chunks exist.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        extracted_path = safe_extract_zip(zip_source, temp_dir)
        return process_repository_dir(
            repo_dir=extracted_path,
            repo_name=repo_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def process_repository_folder(
    uploaded_files: list,
    repo_name: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> Tuple[TFIDFRetriever, RepoSummary]:
    """Safely writes and processes an uploaded repository folder.

    Args:
        uploaded_files: List of uploaded file objects representing folder contents.
        repo_name: Clean display name for the repository.
        chunk_size: Number of lines per chunk.
        overlap: Overlapping lines between consecutive chunks.

    Returns:
        Tuple containing the fitted TFIDFRetriever and a RepoSummary.

    Raises:
        SecurityError: If any file path attempts path traversal, absolute path,
                       Windows drive syntax, contains null bytes, or duplicates.
        ValueError: If folder contains no files, or no supported source files/chunks.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dest_path = safe_write_folder_files(uploaded_files, temp_dir)
        return process_repository_dir(
            repo_dir=dest_path,
            repo_name=repo_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def compute_files_upload_key(file_items: list) -> str:
    """Computes a deterministic content-based cache key for an individual-files upload.

    Uses length-prefixed serialization (same scheme as compute_folder_upload_key) keyed
    with 'mode:files' so the resulting key is always distinct from folder and ZIP keys.

    Args:
        file_items: Collection of uploaded file objects or (rel_path, content_bytes) tuples.

    Returns:
        Deterministic identifier string prefixed with 'files_'.
    """
    extracted_items = []
    for item in file_items:
        if isinstance(item, tuple):
            rel_path, content = item
        else:
            rel_path = item.name
            content = item.getvalue() if hasattr(item, "getvalue") else item.read()

        norm_path = rel_path.replace("\\", "/")
        extracted_items.append((norm_path, content))

    # Sort deterministically by normalized relative path
    extracted_items.sort(key=lambda x: x[0])

    hasher = hashlib.sha256()
    hasher.update(b"mode:files\n")

    for norm_path, content in extracted_items:
        path_bytes = norm_path.encode("utf-8")
        hasher.update(len(path_bytes).to_bytes(8, byteorder="big"))
        hasher.update(path_bytes)
        hasher.update(len(content).to_bytes(8, byteorder="big"))
        hasher.update(content)

    return f"files_{hasher.hexdigest()}"


def process_repository_files(
    file_items: list,
    repo_name: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> Tuple[TFIDFRetriever, RepoSummary]:
    """Stages and processes a collection of individually uploaded source files.

    Each file is written to a temporary directory using its .name attribute as a
    relative path. All existing security checks in safe_write_folder_files() apply:
    path traversal, absolute paths, null bytes, duplicates, and collisions are all
    rejected.

    Streamlit limitation: when files are uploaded via accept_multiple_files=True
    (not "directory"), Streamlit provides only the bare filename with no directory
    structure. Nested relative paths are only available when accept_multiple_files
    is set to "directory". This function accepts both formats safely.

    Args:
        file_items: List of uploaded file objects with .name and .getvalue(), or
                    (rel_path, content_bytes) tuples for testing.
        repo_name: Display name for the repository (not used for indexing).
        chunk_size: Number of lines per chunk.
        overlap: Overlapping lines between consecutive chunks.

    Returns:
        Tuple containing the fitted TFIDFRetriever and a RepoSummary.

    Raises:
        SecurityError: If any file path is unsafe.
        ValueError: If file_items is empty, or no supported files/chunks exist.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dest_path = safe_write_folder_files(file_items, temp_dir)
        return process_repository_dir(
            repo_dir=dest_path,
            repo_name=repo_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def _derive_zip_repo_name(filename: str) -> str:
    """Returns the ZIP filename without its extension."""
    return Path(filename).stem or "uploaded_repo"


def _derive_folder_repo_name(uploaded_files: list) -> str:
    """Derives the display name from the root directory of the uploaded folder files.

    For directory uploads, Streamlit sets .name to the full path within the selected
    directory (e.g. 'my_project/src/main.py' or 'my_project\\src\\main.py').

    Handles missing, absolute, or malformed paths safely. If no directory component
    is available, defaults to 'uploaded_folder'.
    """
    if not uploaded_files:
        return "uploaded_folder"

    cleaned_files = []
    for f in uploaded_files:
        raw_name = getattr(f, "name", "") or ""
        if not isinstance(raw_name, str):
            continue
        cleaned = raw_name.replace("\\", "/").strip()
        while cleaned.startswith("/"):
            cleaned = cleaned[1:]
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if cleaned:
            cleaned_files.append((cleaned, raw_name))

    cleaned_files.sort(key=lambda x: x[0])
    for cleaned, raw_name in cleaned_files:
        parts = [p for p in cleaned.split("/") if p and p != "." and p != ".."]
        if len(parts) >= 2:
            return parts[0]
        if len(parts) == 1 and raw_name.replace("\\", "/").strip().endswith("/"):
            return parts[0]

    return "uploaded_folder"


def parse_unified_upload(raw_upload: Any) -> list:
    """Parses raw upload input from custom component or file objects into a standardized list.

    Supports:
      - None / empty -> []
      - Single file object with .name and (.getvalue() or .read())
      - List of file objects
      - List of dicts: [{'name': str, 'content': base64_str | bytes, 'size': int}, ...]
      - Dict with {'files': [...]}
      - List of (rel_path, content_bytes) tuples
    """
    if not raw_upload:
        return []

    if isinstance(raw_upload, dict) and "files" in raw_upload:
        raw_upload = raw_upload["files"]

    if not isinstance(raw_upload, (list, tuple)):
        raw_items = [raw_upload]
    else:
        raw_items = list(raw_upload)

    parsed = []
    for item in raw_items:
        if isinstance(item, dict):
            name = item.get("name", "")
            raw_content = item.get("content", b"")
            if isinstance(raw_content, str):
                try:
                    content_bytes = base64.b64decode(raw_content)
                except Exception:
                    content_bytes = raw_content.encode("utf-8")
            elif isinstance(raw_content, (bytes, bytearray)):
                content_bytes = bytes(raw_content)
            else:
                content_bytes = b""
            parsed.append(UploadedItem(name=name, data=content_bytes))
        elif hasattr(item, "name") and (hasattr(item, "getvalue") or hasattr(item, "read")):
            parsed.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            parsed.append(UploadedItem(name=item[0], data=item[1]))

    return parsed


def compute_unified_upload_key(items: list) -> str:
    """Computes a deterministic content key for any collection of uploaded items."""
    if not items:
        return ""

    zip_items = [
        item for item in items
        if getattr(item, "name", "").lower().endswith(".zip")
        and "/" not in getattr(item, "name", "").replace("\\", "/").strip("/")
    ]

    if len(items) == 1 and zip_items:
        zip_item = items[0]
        content = zip_item.getvalue() if hasattr(zip_item, "getvalue") else zip_item.read()
        return compute_upload_key(zip_item.name, content)

    has_directories = any(
        "/" in getattr(item, "name", "").replace("\\", "/").strip("/")
        for item in items
    )

    if has_directories:
        return compute_folder_upload_key(items)

    return compute_files_upload_key(items)


def classify_and_process_upload(
    items: list,
    chunk_size: int = 50,
    overlap: int = 10,
) -> Tuple[TFIDFRetriever, RepoSummary, str]:
    """Automatically classifies uploaded items, computes a content hash, and processes them.

    Classification rules:
      - Exactly 1 top-level file ending in .zip -> process_repository_archive()
      - Multiple files containing any top-level .zip -> ValueError (ambiguous input)
      - Non-zip files where at least one file has a directory separator ('/') -> process_repository_folder()
      - Non-zip file(s) where all files are flat with no directory separator -> process_repository_files()

    Returns:
      Tuple of (TFIDFRetriever, RepoSummary, cache_key)
    """
    if not items:
        raise ValueError("The uploaded codebase is empty.")

    zip_items = [
        item for item in items
        if getattr(item, "name", "").lower().endswith(".zip")
        and "/" not in getattr(item, "name", "").replace("\\", "/").strip("/")
    ]

    if zip_items:
        if len(items) > 1 or len(zip_items) > 1:
            raise ValueError(
                "Ambiguous upload: cannot mix a ZIP archive with other files or upload "
                "multiple ZIP archives simultaneously. Please upload either a single ZIP archive, "
                "a repository folder, or individual source files."
            )
        zip_item = items[0]
        content = zip_item.getvalue() if hasattr(zip_item, "getvalue") else zip_item.read()
        repo_name = _derive_zip_repo_name(zip_item.name)
        cache_key = compute_upload_key(zip_item.name, content)
        retriever, summary = process_repository_archive(
            zip_source=io.BytesIO(content),
            repo_name=repo_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return retriever, summary, cache_key

    has_directories = any(
        "/" in getattr(item, "name", "").replace("\\", "/").strip("/")
        for item in items
    )

    if has_directories:
        repo_name = _derive_folder_repo_name(items)
        cache_key = compute_folder_upload_key(items)
        retriever, summary = process_repository_folder(
            uploaded_files=items,
            repo_name=repo_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return retriever, summary, cache_key

    repo_name = "Uploaded files"
    cache_key = compute_files_upload_key(items)
    retriever, summary = process_repository_files(
        file_items=items,
        repo_name=repo_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return retriever, summary, cache_key


def main() -> None:
    """Streamlit application UI entry point."""
    st.set_page_config(
        page_title="Context-Aware Codebase Analyzer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Context-Aware Codebase Analyzer")
    st.caption(
        "Upload a repository to index and search through its source code snippets."
    )

    # ------------------------------------------------------------------ Sidebar
    st.sidebar.header("📁 Upload Codebase")

    # Single unified codebase drop-and-browse area
    raw_upload = _unified_uploader_component(key="codebase_uploader")
    items = parse_unified_upload(raw_upload)

    if not items:
        # No upload active: clear stale index from session state
        st.session_state.pop("active_file_key", None)
        st.session_state.pop("retriever", None)
        st.session_state.pop("repo_summary", None)
    else:
        try:
            preview_key = compute_unified_upload_key(items)
            if st.session_state.get("active_file_key") != preview_key:
                st.session_state.pop("active_file_key", None)
                st.session_state.pop("retriever", None)
                st.session_state.pop("repo_summary", None)

                with st.sidebar.status("Processing codebase…", expanded=True) as status:
                    retriever, summary, key = classify_and_process_upload(items)
                    st.session_state["retriever"] = retriever
                    st.session_state["repo_summary"] = summary
                    st.session_state["active_file_key"] = key
                    status.update(label="Indexing complete!", state="complete", expanded=False)
        except (ExtractionError, SecurityError, ValueError) as err:
            st.session_state.pop("active_file_key", None)
            st.session_state.pop("retriever", None)
            st.session_state.pop("repo_summary", None)
            st.sidebar.error(str(err))
        except Exception as err:
            st.session_state.pop("active_file_key", None)
            st.session_state.pop("retriever", None)
            st.session_state.pop("repo_summary", None)
            st.sidebar.error(f"An unexpected error occurred: {err}")

    # ------------------------------------------------ Repository statistics panel
    summary: Optional[RepoSummary] = st.session_state.get("repo_summary")
    retriever: Optional[TFIDFRetriever] = st.session_state.get("retriever")

    if summary is not None:
        st.sidebar.divider()
        st.sidebar.subheader(f"📊 {summary.repo_name}")
        st.sidebar.metric("Supported Files", summary.total_files)
        st.sidebar.metric("Indexed Chunks", summary.total_chunks)

        if summary.language_breakdown:
            st.sidebar.markdown("**Languages:**")
            for lang, count in sorted(
                summary.language_breakdown.items(), key=lambda item: -item[1]
            ):
                st.sidebar.write(f"- `{lang}`: {count} file{'s' if count > 1 else ''}")

        with st.sidebar.expander("Indexed Files", expanded=False):
            for path in summary.file_list:
                st.caption(f"📄 {path}")

    # ---------------------------------------------------------- Main search area
    if retriever is None or not retriever.is_indexed:
        st.info("👈 Upload a repository ZIP, folder, or individual files in the sidebar to get started.")
        return

    st.subheader("🔎 Search Codebase")
    col_query, col_k = st.columns([4, 1])

    with col_query:
        query = st.text_input(
            "Natural language question or search query:",
            placeholder="e.g. Where is database authentication configured?",
            key="query_input",
        )

    with col_k:
        top_k = st.number_input(
            "Top K results:",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            key="top_k_input",
        )

    if query.strip():
        results = retriever.search(query=query.strip(), top_k=int(top_k))

        if not results:
            st.warning("No matching code snippets found for this query.")
        else:
            st.success(
                f"Found {len(results)} relevant snippet{'s' if len(results) > 1 else ''}:"
            )
            for i, result in enumerate(results, start=1):
                chunk = result.chunk
                title = (
                    f"**#{i}** `{chunk.file_path}` "
                    f"(Lines {chunk.start_line}–{chunk.end_line}) — "
                    f"Similarity: `{result.score:.4f}`"
                )
                with st.expander(title, expanded=(i <= 3)):
                    st.caption(
                        f"Language: **{chunk.language}** | Chunk ID: `{chunk.chunk_id}`"
                    )
                    st.code(chunk.content, language=chunk.language)
    else:
        st.caption("Enter a query above to search through indexed code snippets.")


if __name__ == "__main__":
    main()
