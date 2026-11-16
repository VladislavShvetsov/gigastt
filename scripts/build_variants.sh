#!/usr/bin/env bash
set -euo pipefail
# Run inside the root of the fork; keep CPU and GPU build output isolated.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v cargo >/dev/null || { echo 'cargo not installed' >&2; exit 1; }
variant="${1:-}"
case "$variant" in
  cpu)
    export CARGO_TARGET_DIR="$ROOT/target-cpu"
    cargo build --release --locked -p gigastt --no-default-features
    echo "CPU executable: $CARGO_TARGET_DIR/release/gigastt"
    ;;
  cuda)
    export CARGO_TARGET_DIR="$ROOT/target-cuda"
    cargo build --release --locked -p gigastt --no-default-features --features cuda
    echo "CUDA executable: $CARGO_TARGET_DIR/release/gigastt"
    echo 'NOTE: runtime requires the ORT rc.14 CUDA provider pair from this build, plus compatible CUDA/cuDNN shared libraries.'
    find "$CARGO_TARGET_DIR/release/build" -type f \( -name 'libonnxruntime_providers_cuda.so' -o -name 'libonnxruntime_providers_shared.so' \) -print || true
    ;;
  *) echo 'Usage: scripts/build_variants.sh cpu|cuda' >&2; exit 2;;
esac
