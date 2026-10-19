#!/usr/bin/env python3
"""Comprueba SHA-256 de los cinco archivos originales de la tesis (stdlib)."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data' / 'manifests' / 'archives_sha256.json'

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def verify(folder: Path, manifest: Path = MANIFEST) -> bool:
    expected = json.loads(manifest.read_text(encoding='utf-8'))
    good = True
    for name, info in expected.items():
        candidates = [folder / name, folder / 'data' / 'archives' / name]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            print(f'MISSING  {name}  (no se ha suministrado)')
            good = False
            continue
        size_ok = source.stat().st_size == int(info['bytes'])
        hash_ok = sha256(source).lower() == info['sha256'].lower()
        state = 'OK     ' if size_ok and hash_ok else 'INVALID'
        print(f'{state}  {name}  ({source.stat().st_size:,} bytes)')
        if not (size_ok and hash_ok):
            good = False
    return good

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path.cwd(),
                        help='Carpeta con los ZIP originales o el clon Git con data/archives')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    return 0 if verify(args.directory.expanduser().resolve(), args.manifest) else 1

if __name__ == '__main__':
    sys.exit(main())
