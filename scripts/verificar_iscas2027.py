#!/usr/bin/env python3
"""Verify ISCAS2027 file SHA-256 and Git LFS pointer OIDs against provenance.

In GitHub Actions checkout without LFS download, binary files remain lightweight
pointer files: matching their OID to the original manifest verifies metadata only.
For independently verifying binary bytes, run git lfs pull first and rerun.
No third-party Python packages or cosmological data are required.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "projects" / "ISCAS2027"
DEFAULT_MANIFEST = PROJECT / "provenance" / "FILES_SHA256.txt"
LINE = re.compile(r"^([0-9a-f]{64})\s+(.+)$")
OID = re.compile(r"^oid sha256:([0-9a-f]{64})$")
SIZE = re.compile(r"^size (\d+)$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(manifest: Path) -> bool:
    if not manifest.is_file():
        print(f"ERROR: missing manifest {manifest}", file=sys.stderr)
        return False
    good = True
    stats = {"exact": 0, "lfs_pointer": 0, "generated_skipped": 0, "failed": 0}
    for line_num, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE.fullmatch(raw.strip())
        if match is None:
            print(f"FAIL invalid manifest line {line_num}", file=sys.stderr)
            stats["failed"] += 1
            good = False
            continue
        expected, relative = match.groups()
        target = (REPO / relative).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            print(f"FAIL path outside project: {relative}", file=sys.stderr)
            stats["failed"] += 1
            good = False
            continue
        if not target.is_file():
            if target.name in {"main.aux", "main.log", "main.synctex.gz"} and "latex" in target.parts:
                stats["generated_skipped"] += 1
                print(f"SKIP generated build artifact: {relative}")
                continue
            print(f"FAIL missing: {relative}", file=sys.stderr)
            stats["failed"] += 1
            good = False
            continue
        if target.is_symlink():
            print(f"FAIL unexpected symlink: {relative}", file=sys.stderr)
            stats["failed"] += 1
            good = False
            continue

        if target.stat().st_size < 300:
            candidate = target.read_bytes()
            if candidate.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                fields = candidate.decode("ascii", errors="strict").splitlines()
                match_oid = next((OID.fullmatch(t) for t in fields if t.startswith("oid ")), None)
                match_size = next((SIZE.fullmatch(t) for t in fields if t.startswith("size ")), None)
                if match_oid and match_oid.group(1) == expected and match_size and int(match_size.group(1)) > 0:
                    stats["lfs_pointer"] += 1
                    print(f"POINTER OK {relative}")
                    continue
                print(f"FAIL LFS pointer: {relative}", file=sys.stderr)
                stats["failed"] += 1
                good = False
                continue

        if digest(target) == expected:
            stats["exact"] += 1
            print(f"SHA256 OK {relative}")
        else:
            print(f"FAIL SHA256: {relative}", file=sys.stderr)
            stats["failed"] += 1
            good = False
    print("Summary:", ", ".join(f"{name}={n}" for name, n in stats.items()))
    if stats["lfs_pointer"]:
        print("Note: LFS POINTER OK checks oid/size metadata, not downloadable binary content.")
    return good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    return 0 if verify(args.manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
