#!/usr/bin/env python3
"""Build an explicit, locally verified GigaAM E2E RNNT FP32 model pack.
Requires: pip install huggingface_hub; Python >= 3.11.
Does not alter the GigaSTT code or the source model files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os
import sys

FILES = {
    'encoder': 'v3_e2e_rnnt_encoder.onnx',
    'decoder': 'v3_e2e_rnnt_decoder.onnx',
    'joint': 'v3_e2e_rnnt_joint.onnx',
    'vocab': 'v3_e2e_rnnt_vocab.txt',
}
MANIFEST = '''# GigaSTT FP32-only GigaAM v3 E2E RNNT model pack.
# Do not add encoder_int8: GigaSTT prefers INT8 when that file exists.
architecture = "e2e_rnnt"

[files]
encoder = "v3_e2e_rnnt_encoder.onnx"
decoder = "v3_e2e_rnnt_decoder.onnx"
joint = "v3_e2e_rnnt_joint.onnx"
vocab = "v3_e2e_rnnt_vocab.txt"
'''

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),''):
            h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('models/gigaam-e2e-fp32'))
    ap.add_argument('--repo',default='istupakov/gigaam-v3-onnx')
    ap.add_argument('--revision',required=True,help='Pinned Hugging Face commit hash (not main)')
    ap.add_argument('--no-download',action='store_true',help='Validate files already present; no network')
    ap.add_argument('--force',action='store_true',help='Overwrite an existing manifest')
    args=ap.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if not args.no_download:
        from huggingface_hub import hf_hub_download
        for name in FILES.values():
            print('Download:',name,flush=True)
            hf_hub_download(repo_id=args.repo,filename=name,revision=args.revision,local_dir=str(output))
    for field,name in FILES.items():
        p=output/name
        if not p.is_file() or p.stat().st_size==0:
            ap.error(f'Missing or empty {field}: {p}')
    # Refuse stray prequantized encoders to make backend choice unambiguous.
    extra=list(output.glob('*encoder_int8.onnx'))
    if extra:
        ap.error('FP32-only pack contains INT8 encoder(s): '+', '.join(map(str,extra)))
    manifest=output/'manifest.toml'
    if manifest.exists() and manifest.read_text(encoding='utf8')!=MANIFEST and not args.force:
        ap.error('Manifest differs; use --force to replace, or review it manually')
    manifest.write_text(MANIFEST,encoding='utf8')
    hashes={name:sha(output/name) for name in FILES.values()}
    (output/'SHA256SUMS').write_text(''.join(f'{hashes[n]}  {n}\n' for n in sorted(hashes)),encoding='ascii')
    metadata={'repo':args.repo,'revision':args.revision,'architecture':'e2e_rnnt','precision':'fp32', 'files':hashes}
    (output/'model-source.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Prepared:',output)
    print('Manifest:',manifest)
    print('Checksums:',output/'SHA256SUMS')
if __name__=='__main__':main()
