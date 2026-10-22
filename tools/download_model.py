#!/usr/bin/env python3
"""Download and verify the pinned von OptionMarker checkpoint.

Downloads `option_marker.pt` from the pinned Hugging Face revision and verifies
its SHA-256 before installing it. This is the first half of model installation;
`tools/convert_option_marker.py` turns it into the verified
`option_marker.safetensors` inference artifact.

Standard library only — no torch or safetensors required to download or verify.
Importing this module reuses the pinned source constants from the converter but
does not import the heavy model libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# Make the sibling converter importable whether this script is run directly or
# loaded by the model-free test harness via importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_option_marker import HF_REPO, HF_REVISION, SOURCE_ARTIFACT_NAME, SOURCE_SHA256

DOWNLOAD_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{SOURCE_ARTIFACT_NAME}"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "weights" / SOURCE_ARTIFACT_NAME
_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verified(path: Path) -> bool:
    return path.is_file() and sha256_file(path) == SOURCE_SHA256


def download(output: Path) -> None:
    """Download to `output` atomically, verifying the pinned SHA-256."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if _is_verified(output):
        print(f"already present and verified: {output}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        print(f"downloading {DOWNLOAD_URL}")
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(_CHUNK_SIZE):
                    handle.write(chunk)

        actual = sha256_file(temporary)
        if actual != SOURCE_SHA256:
            raise SystemExit(f"SHA256 mismatch: {actual} != {SOURCE_SHA256}")

        os.replace(temporary, output)
        print(f"verified and installed: {output} (sha256={actual})")
    finally:
        temporary.unlink(missing_ok=True)


def check(output: Path) -> int:
    """Verify an existing file without downloading; return an exit code."""
    if not output.is_file():
        print(f"missing: {output}", file=sys.stderr)
        return 1
    actual = sha256_file(output)
    if actual != SOURCE_SHA256:
        print(f"SHA256 mismatch: {actual} != {SOURCE_SHA256}", file=sys.stderr)
        return 1
    print(f"verified: {output} (sha256={actual})")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"destination path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing file without downloading",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        return check(args.output)
    download(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
