"""Verify exported source bytes, optionally against an original source tree."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance.json').read_text())
    for row in manifest['files']:
        path = root / row['release_path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise SystemExit(f"Export changed: {row['release_path']}")
        if args.source_root:
            actual_source = hashlib.sha256((args.source_root / row['source_path']).read_bytes()).hexdigest()
            expected = row.get('source_sha256', row['sha256'])
            if actual_source != expected:
                raise SystemExit(f"Original source changed since extraction: {row['source_path']}")
    print(f"Verified {len(manifest['files'])} exported source records")


if __name__ == '__main__':
    main()
