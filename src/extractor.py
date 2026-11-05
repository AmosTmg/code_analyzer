"""Safe ZIP extraction module with robust path traversal protection."""

import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import BinaryIO, Union


class ExtractionError(Exception):
    """Base exception for archive extraction failures."""
    pass


class InvalidZipError(ExtractionError):
    """Raised when the provided file is not a valid or readable ZIP archive."""
    pass


class SecurityError(ExtractionError):
    """Raised when a security risk (e.g., Zip Slip, path traversal, symlink) is detected."""
    pass


class EmptyArchiveError(ExtractionError):
    """Raised when the archive is empty or contains no extractable files."""
    pass


def safe_extract_zip(
    zip_source: Union[str, Path, BinaryIO],
    destination_dir: Union[str, Path]
) -> Path:
    """Safely extracts a ZIP archive to the target destination directory.

    Guards against:
      - Zip Slip / path traversal attacks (e.g. '../path/file', '/absolute/path')
      - Malicious symlinks embedded in archives
      - Corrupted or non-ZIP files
      - Empty archives

    Args:
        zip_source: Path to the ZIP file or a readable binary file-like object (e.g. BytesIO).
        destination_dir: Directory where files should be extracted.

    Returns:
        Path: The canonical resolved Path to the extraction directory.

    Raises:
        InvalidZipError: If the input cannot be read as a valid ZIP archive.
        SecurityError: If an entry attempts path traversal or introduces symlinks.
        EmptyArchiveError: If the archive contains no files.
    """
    dest_path = Path(destination_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_source, mode="r") as zf:
            members = zf.infolist()
            if not members:
                raise EmptyArchiveError("The provided ZIP archive is empty.")

            # Filter out and count file entries
            file_entries = [m for m in members if not m.is_dir()]
            if not file_entries:
                raise EmptyArchiveError("The archive contains directories but no extractable files.")

            # Pass 1: Security validation
            for member in members:
                filename = member.filename

                # Reject absolute path strings
                if os.path.isabs(filename) or filename.startswith("/") or filename.startswith("\\"):
                    raise SecurityError(
                        f"Absolute path detected in archive member: '{filename}'"
                    )

                # Check for symlink attribute (external_attr >> 16)
                mode = member.external_attr >> 16
                if mode and stat.S_ISLNK(mode):
                    raise SecurityError(
                        f"Symbolic link detected in archive member: '{filename}'"
                    )

                # Resolve target path and verify it stays strictly inside dest_path
                target_path = (dest_path / filename).resolve()
                try:
                    common = os.path.commonpath([str(dest_path), str(target_path)])
                except ValueError as err:
                    raise SecurityError(
                        f"Cross-device or invalid path traversal detected: '{filename}'"
                    ) from err

                if common != str(dest_path):
                    raise SecurityError(
                        f"Path traversal detected in archive member: '{filename}'"
                    )

            # Pass 2: Controlled extraction
            for member in members:
                target_path = (dest_path / member.filename).resolve()
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

            return dest_path

    except (zipfile.BadZipFile, zipfile.LargeZipFile) as err:
        raise InvalidZipError(f"Corrupted or invalid ZIP archive: {err}") from err
