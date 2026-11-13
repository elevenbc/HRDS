"""Validate the release inventory and Python syntax without GPU dependencies."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        p = ROOT / name
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != info['sha256']:
            raise SystemExit('Hash mismatch: ' + name)
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=name)
    print('Verified {} files and Python syntax.'.format(len(manifest['files'])))
    print('Core files match the recorded paper-run hashes:', manifest['core_hash_match'])


if __name__ == '__main__':
    main()
