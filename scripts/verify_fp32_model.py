#!/usr/bin/env python3
"""Offline integrity and manifest validation; Python >= 3.11."""
import argparse
import hashlib
from pathlib import Path
import tomllib
import json

REQUIRED=('encoder','decoder','joint','vocab')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('dir',type=Path)
    args=parser.parse_args()
    root=args.dir.resolve()
    m=tomllib.loads((root/'manifest.toml').read_text(encoding='utf8'))
    assert m['architecture']=='e2e_rnnt', 'Unexpected architecture'
    assert m.get('precision')=='fp32', 'Explicit FP32 precision is required'
    assert 'encoder_int8' not in m['files'], 'INT8 priority would change effective model'
    sums={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        sums[name.strip()]=expected.lower()
    for field in REQUIRED:
        name=m['files'][field]
        assert Path(name).name==name and '..' not in name, f'Unsafe basename: {name}'
        path=root/name
        assert path.is_file() and path.stat().st_size>0, f'Missing {path}'
        assert sums.get(name)==sha(path), f'Checksum mismatch: {path}'
        print(f'OK {field}: {name}')
    src=root/'model-source.json'
    if src.exists():
        metadata=json.loads(src.read_text())
        assert metadata['architecture']=='e2e_rnnt'
        for name in m['files'].values():
            assert metadata['files'].get(name)==sums[name],f'Metadata hash mismatch: {name}'
    print('FP32-only model pack: VERIFIED')
if __name__=='__main__':main()
