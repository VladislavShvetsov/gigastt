#!/usr/bin/env python3
"""Apply two small, source-anchored FP32 loader fixes to fork checkout.
Refuses unknown code layouts; back up with git before running.
"""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]

p=root/'crates/gigastt-core/src/inference/load_files.rs'
s=p.read_text()
a='''        if let Some(m) = ModelManifest::load(dir)? {
            // GIGASTT_FP32_MANIFEST_V1: allow FP32 in an explicit model pack.
            let encoder = m.preferred_encoder_path(dir);
            anyhow::ensure!(encoder.is_file(), "encoder not found: {}", encoder.display());
            return Ok(Self {
                encoder: m.preferred_encoder_path(dir),
                decoder: m.decoder_path(dir),
                joint: m.joint_path(dir),
                vocab: m.vocab_path(dir),
                using_int8: m.prefers_int8(dir),
                encoder_hash: None,
            });
        }'''
b='''        if let Some(m) = ModelManifest::load(dir)? {
            anyhow::ensure!(
                m.architecture == variant,
                "manifest architecture '{}' conflicts with selected model variant '{}'",
                m.architecture.as_str(), variant.as_str()
            );
            let encoder = m.preferred_encoder_path(dir);
            let decoder = m.decoder_path(dir);
            let joint = m.joint_path(dir);
            let vocab = m.vocab_path(dir);
            anyhow::ensure!(encoder.is_file(), "encoder not found: {}", encoder.display());
            anyhow::ensure!(vocab.is_file(), "vocab not found: {}", vocab.display());
            if !variant.is_ctc() {
                anyhow::ensure!(decoder.as_ref().is_some_and(|p| p.is_file()), "decoder file missing");
                anyhow::ensure!(joint.as_ref().is_some_and(|p| p.is_file()), "joint file missing");
            }
            return Ok(Self {
                encoder,
                decoder,
                joint,
                vocab,
                using_int8: m.prefers_int8(dir),
                encoder_hash: None,
            });
        }'''
if a in s:
    p.write_text(s.replace(a,b,1))
    print('Updated',p)
elif b in s:
    print('Already patched',p)
else:
    sys.exit('Unknown source layout: refused load_files.rs modification')

p=root/'crates/gigastt-core/src/model/download.rs'
s=p.read_text()
a='''    if is_prequantized_present(variant, dir) {
        tracing::info!("Using existing {variant:?} INT8 model at {model_dir}");
        return Ok(variant);
    }'''
b='''    if is_usable_present(variant, dir) {
        tracing::info!("Using existing {variant:?} model at {model_dir}");
        return Ok(variant);
    }'''
if a in s:
    p.write_text(s.replace(a,b,1))
    print('Updated',p)
elif b in s:
    print('Already patched',p)
else:
    sys.exit('Unknown source layout: refused download.rs modification (first check)')

# The post-lock check must use the same FP32-aware presence predicate.
s=p.read_text()
a='''    if is_prequantized_present(variant, dir) {
        tracing::info!("Pre-quantized {variant:?} model found at {model_dir} after lock");
        return Ok(variant);
    }'''
b='''    if is_usable_present(variant, dir) {
        tracing::info!("{variant:?} model found at {model_dir} after lock");
        return Ok(variant);
    }'''
if a in s:
    p.write_text(s.replace(a,b,1))
    print('Updated post-lock',p)
elif b in s:
    print('Already patched post-lock',p)
else:
    sys.exit('Unknown source layout: refused download.rs modification (post-lock)')
