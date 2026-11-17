# GigaSTT

Локальное распознавание речи GigaAM: CLI и Rust-библиотека, CPU/NVIDIA CUDA 13, streaming, диаризация и INT8-квантование.

Требования: Rust 1.94+, C/C++ toolchain, `protoc`. Для GPU — CUDA 13 и совместимый cuDNN.

Сборка: `cargo build --release`. Для GPU добавьте `--features cuda`. Бинарник: `target/release/gigastt` (`.exe` на Windows).

CPU:

```sh
cargo run --release -- download --model-dir models
cargo run --release -- --execution-provider cpu transcribe audio.wav --model-dir models
```

CUDA 13:

```sh
cargo run --release --features cuda -- download --fp32 --model-dir models-fp32
cargo run --release --features cuda -- --execution-provider cuda transcribe audio.wav --model-dir models-fp32
```

Также доступны `transcribe-batch`, `watch`, `quantize`, `cache-gc`. Параметры: `gigastt <команда> --help`.

Лицензия: MIT. Сторонние компоненты: `NOTICE`.
