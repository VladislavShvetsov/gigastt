# GigaSTT FP32 CUDA / CPU — deployment guide

This guide targets the fork branch `cuda-fp32-blackwell`. The fork already implements manifest-based model loading and upgrades Rust `ort` to 2.0.0-rc.14. These are **build-from-source instructions**; model weights are downloaded separately and are not committed to Git.

## Prerequisites

- Linux x86_64, Rust toolchain, `protobuf-compiler`, C compiler, `pkg-config`, SSL development headers.
- Python >=3.11 for model pack preparation; `pip install huggingface_hub` or `uv pip install huggingface_hub`.
- NVIDIA driver and compatible CUDA/cuDNN libraries for CUDA inference. CPU build needs neither.

## Prepare an immutable FP32-only model pack

Pin the *actual* revision (40-hex commit) from `https://huggingface.co/istupakov/gigaam-v3-onnx/tree/main` rather than using a moving branch:

```bash
python3 scripts/prepare_fp32_model.py \
  --revision YOUR_VERIFIED_40_HEX_HF_COMMIT \
  --output /srv/models/gigaam-e2e-fp32
python3 scripts/verify_fp32_model.py /srv/models/gigaam-e2e-fp32
```

For a pre-provisioned offline model directory, use `--no-download --revision YOUR_VERIFIED_40_HEX_HF_COMMIT`; provenance in this mode is **user-declared**, not authenticated by the script. Verify against external trusted checksums.

The generator writes exactly:

```toml
architecture = "e2e_rnnt"
[files]
encoder = "v3_e2e_rnnt_encoder.onnx"
decoder = "v3_e2e_rnnt_decoder.onnx"
joint = "v3_e2e_rnnt_joint.onnx"
vocab = "v3_e2e_rnnt_vocab.txt"
```

**Do not add `encoder_int8`**: the current fork manifest loader selects that encoder whenever it exists. This is not a precision toggle.

`SHA256SUMS` protects against accidental changes *after* preparation, not malicious substitution at the original download. Pin the HF commit; ideally compare checksums with trusted artifact manifests.

## Harden manifest loading (one-time source patch)

Run against the checked-out fork branch, before compiling:

```bash
python3 scripts/harden_fp32_loader.py
git diff --check
```

This enforces selected-architecture consistency, checks all required manifest files before opening ONNX sessions, and teaches the model ensure/download path to accept a complete FP32 manifest pack instead of starting an INT8 download. The script refuses to patch unexpected source code. Review its diff and commit the resulting Rust changes.

## Build

```bash
bash scripts/build_variants.sh cpu
bash scripts/build_variants.sh cuda
```

The builds use isolated targets `target-cpu` and `target-cuda` to prevent publishing the wrong executable. The CUDA build downloads a native ONNX Runtime binary matching `ort rc.14` and the platform; the native runtime and dynamic providers are not tracked in Cargo.lock. **Never mix provider libraries from Python onnxruntime with the Rust binary.**

## Runtime smoke test

```bash
export GIGASTT_OFFLINE=1
export GIGASTT_EXECUTION_PROVIDER=cuda  # or cpu
export RUST_LOG=info
# Set LD_LIBRARY_PATH to compatible NVIDIA CUDA/cuDNN runtime libraries.
# Put both ONNX Runtime CUDA provider .so files from the same native binary build next to the executable.
./target-cuda/release/gigastt transcribe /srv/test/short.wav \
  --model-dir /srv/models/gigaam-e2e-fp32 \
  --model-variant e2e_rnnt \
  --punctuation off --itn off --format json \
  --output /srv/test/fp32-gpu.json
```

**Acceptance:** logs explicitly show FP32 encoder + CUDA EP; no fallback; output contains plausible transcribed words. Inspect `nvidia-smi` to verify GPU use. For CPU, use `target-cpu/release/gigastt` and `GIGASTT_EXECUTION_PROVIDER=cpu`.

The `CUDA provider` prebuilt artifacts are specific to Rust `ort` and native CUDA versions. Standard hosted GitHub runners have no NVIDIA GPU, so `cuda-compile` checks compilation **only**. For production, run the smoke test on a separate GPU self-hosted runner and compare word-level output with a frozen reference recording.

## Production acceptance gates

- Freeze fork commit, Cargo.lock, Hugging Face revision, SHA256SUMS, OS and driver constraints.
- CPU binary builds and transcribes the same model pack.
- CUDA binary builds, links matching `libonnxruntime_providers_cuda.so` + `libonnxruntime_providers_shared.so` and transcribes on RTX 5090.
- FP32 is confirmed through logs, not assumed from filename.
- Existing long-file pipeline regression: word count, word timestamps, overlap stitching, and speaker attribution against known reference output.
- Do not ship the locally written Python transcription/diarization orchestrator as part of the upstream GigaSTT fork; keep it in the `stt-engine` repository.

## Known code issue to review before shipping

The hardening script adds variant checks and early required-file presence checks; add Rust unit tests for conflicting `--model-variant` before merging.

Update outdated INT8-only function documentation. Confirm the CLI goes through `ensure_prequantized_model_variant` with an FP32-only manifest pack without attempting an INT8 download.
