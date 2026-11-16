# GigaSTT

Локальный движок распознавания речи GigaAM: CPU и NVIDIA CUDA 13, streaming, диаризация, INT8-квантование.

Требования: Rust 1.94+, C/C++ toolchain, `protoc`. Для GPU — CUDA 13 и совместимый cuDNN.

```sh
cargo build --release -p gigastt-core
cargo build --release -p gigastt-core --features cuda
```

Загрузка FP32-модели для GPU:

```sh
python -m pip install huggingface_hub
python scripts/prepare_fp32_model.py --revision <commit>
python scripts/verify_fp32_model.py models/gigaam-e2e-fp32
```

API: `gigastt_core::inference::Engine`. Устройство: `ExecutionProviderChoice::{Cpu, Cuda, Auto}`. INT8-модели: `model::ensure_model_variant`.

Лицензия: MIT. Сторонние компоненты: `NOTICE`.
