"""Read verified historical docs/research from archives without recreating old roots."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class MissingLocalArchive(FileNotFoundError):
    """The optional local evidence archive is not installed."""


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=2)
def _archive(kind):
    base = ROOT / '.vscode' / kind
    manifest_path = base / 'archive-manifest.json'
    if not manifest_path.is_file():
        raise MissingLocalArchive(f'local evidence is unavailable: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    archive = base / manifest['archive']
    if not archive.is_file():
        raise MissingLocalArchive(f'local evidence is unavailable: {archive}')
    if _sha(archive) != manifest['archive_sha256']:
        raise ValueError(f'archive checksum mismatch: {archive}')
    return archive, manifest


def archived_path(relative):
    """Return a verified scratch copy of one historical file or directory."""
    name = PurePosixPath(str(relative))
    if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] not in ('docs', 'research'):
        raise ValueError('expected a relative docs/ or research/ archive path')
    key = name.as_posix()
    archive, manifest = _archive(name.parts[0])
    selected = {p:info for p,info in manifest['files'].items() if p == key or p.startswith(key + '/')}
    if not selected:
        raise FileNotFoundError(key)
    cache = ROOT / 'scratch' / 'document-archives' / manifest['archive_sha256']
    with tarfile.open(archive) as tar:
        for member, info in selected.items():
            path = cache / member
            if path.is_file() and _sha(path) == info['sha256']:
                continue
            entry = tar.getmember(member)
            if not entry.isfile():
                raise ValueError(f'not a regular archived file: {member}')
            data = tar.extractfile(entry).read()
            if hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError(f'archived file checksum mismatch: {member}')
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                stream.write(data)
                temporary = stream.name
            os.replace(temporary, path)
    return cache / key


def enable_archived_research():
    """Make preserved research modules available to regression tests only."""
    from unittest import SkipTest

    try:
        parent = archived_path('research').parent
    except MissingLocalArchive as error:
        raise SkipTest(str(error)) from error
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='original docs/ or research/ path')
    args = parser.parse_args()
    print(archived_path(args.path))


if __name__ == '__main__':
    main()
