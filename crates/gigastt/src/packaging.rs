//! `download` / `quantize` / `cache-gc` command bodies.

use gigastt_core::model;
use gigastt_core::model::{ModelVariant, ProgressMode};

fn ensure_fp32_manifest(model_dir: &str, variant: ModelVariant) -> anyhow::Result<()> {
    use std::io::Write;
    anyhow::ensure!(
        variant == ModelVariant::E2eRnnt,
        "--fp32 requires --model-variant e2e_rnnt"
    );
    let dir = std::path::Path::new(model_dir);
    let text = format!(
        "architecture = \"e2e_rnnt\"\nprecision = \"fp32\"\n\n[files]\nencoder = \"{}\"\ndecoder = \"{}\"\njoint = \"{}\"\nvocab = \"{}\"\n",
        variant.encoder_file(),
        variant.decoder_file(),
        variant.joint_file(),
        variant.vocab_file(),
    );
    std::fs::create_dir_all(dir)?;
    let path = dir.join(model::MANIFEST_FILE);
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => file.write_all(text.as_bytes())?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read_to_string(&path)?;
            let config: toml::Value = toml::from_str(&existing)?;
            anyhow::ensure!(
                config.get("precision").and_then(toml::Value::as_str) == Some("fp32")
                    && model::ModelManifest::parse(&existing)?
                        == model::ModelManifest::parse(&text)?,
                "existing manifest differs from the FP32 E2E pack; use a separate --model-dir"
            );
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

/// Packaging: rebuild INT8 encoder from a local FP32 ONNX (no download).
pub(crate) fn run_quantize(
    model_dir: String,
    force: bool,
    quantize_conv: bool,
    requested: Option<ModelVariant>,
) -> anyhow::Result<()> {
    // Rebuild only with explicit replacement; an existing file may use a
    // different recipe, so a no-op must not claim successful conversion.
    let dir = std::path::Path::new(&model_dir);
    let resolved = requested
        .or_else(|| model::ModelVariant::detect_in_dir(dir))
        .unwrap_or_default();
    let input = dir.join(resolved.encoder_file());
    let output = dir.join(resolved.encoder_int8_file());
    if output.exists() && !force {
        anyhow::bail!(
            "INT8 encoder already exists at {}. Use --force to rebuild it.",
            output.display()
        );
    }
    if !input.is_file() {
        anyhow::bail!(
            "FP32 encoder not found at {} — `quantize` needs the FP32 ONNX as packaging source. \
             For runtime, use `gigastt download` (lean INT8 only).",
            input.display()
        );
    }
    if !quantize_conv {
        tracing::info!("Leaving Conv in FP32; quantizing MatMul and Gemm only");
    }
    let ops = gigastt_core::quantize::QuantizeOps {
        conv: quantize_conv,
    };
    gigastt_core::quantize::quantize_model_with(&input, &output, ops)?;
    tracing::info!("Quantized model saved to {}", output.display());
    Ok(())
}

pub(crate) fn run_cache_gc(
    model_dir: String,
    dry_run: bool,
    dedupe: bool,
    optimized_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let dir = std::path::Path::new(&model_dir);
    let prune = match optimized_cache_dir {
        Some(cache_dir) => {
            model::prune_optimized_cache_dir(std::path::Path::new(&cache_dir), dir, dry_run)?
        }
        None => model::prune_optimized_cache(dir, dry_run)?,
    };
    let action = if dry_run { "would free" } else { "freed" };
    println!(
        "optimized_cache: kept {} graph(s), removed {} ({} {:.1} MiB)",
        prune.kept.len(),
        prune.removed.len(),
        action,
        prune.freed_bytes as f64 / (1024.0 * 1024.0),
    );
    for p in &prune.removed {
        println!("  - {}", p.display());
    }
    if dedupe {
        let d = model::dedupe_model_dir(dir, dry_run)?;
        println!(
            "dedupe: {} group(s), {} hardlink(s), {} {:.1} MiB",
            d.groups,
            d.hardlinked,
            action,
            d.freed_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

/// Download the lean INT8 model (and optional side assets). On failure or
/// SIGINT exits the process with a progress-kind exit code (never returns `Err`
/// for those paths).
pub(crate) async fn run_download(
    model_dir: String,
    model_variant: ModelVariant,
    fp32: bool,
    #[cfg(feature = "diarization")] skip_diarization: bool,
    progress: ProgressMode,
) -> anyhow::Result<()> {
    model::set_progress_mode(progress);
    if fp32 {
        ensure_fp32_manifest(&model_dir, model_variant)?;
    }
    // `download` is an explicit action: the requested variant maps to
    // the default (Rnnt) so a bare `gigastt download` fetches something
    // useful. **INT8 only** — lean prequantized bundle (or CTC INT8 from HF).
    //
    // The flow runs on its own task: large-file SHA-256 verify is
    // synchronous, and polled inline it would starve the select's signal
    // branch — Ctrl-C must interrupt immediately (sidecar cancel path).
    let dl_model_dir = model_dir.clone();
    let mut download = tokio::spawn(async move {
        let model_dir = dl_model_dir;
        if fp32 {
            model::ensure_fp32_model_variant(Some(model_variant), &model_dir).await?;
        } else if model_variant.is_ctc() {
            // CTC heads: HF pre-quantized INT8 encoder (+ vocab) directly.
            model::ensure_model_variant(Some(model_variant), &model_dir).await?;
        } else {
            // RNN-T: lean INT8 bundle from the pinned Release.
            model::ensure_prequantized_model_variant(Some(model_variant), &model_dir).await?;
        }
        #[cfg(feature = "diarization")]
        {
            if !skip_diarization {
                model::ensure_speaker_model(&model_dir).await?;
            }
        }

        tracing::info!("Model ready at {model_dir}");
        anyhow::Ok(())
    });
    // Resolves only on a *delivered* SIGINT. A failed handler
    // registration is logged and parks forever — it must not fabricate
    // an interrupt and abort a healthy download with exit 130.
    let interrupted = async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => (),
            Err(e) => {
                tracing::warn!("Failed to listen for Ctrl-C: {e}");
                std::future::pending::<()>().await
            }
        }
    };
    tokio::select! {
        joined = &mut download => {
            // Flatten the JoinHandle: a panic inside the download task
            // is reported through the same error contract.
            let result = joined
                .map_err(|e| anyhow::anyhow!("download task failed: {e}"))
                .and_then(|r| r);
            match result {
                Ok(()) => {
                    model::emit_progress_event(&model::ProgressEvent::Done {
                        model_dir: model_dir.clone(),
                    });
                }
                Err(e) => {
                    let kind = model::classify_download_error(&e);
                    model::emit_progress_event(&model::ProgressEvent::Error {
                        kind,
                        message: format!("{e:#}"),
                    });
                    // Same rendering anyhow's `Termination` would print,
                    // then the documented per-kind exit code (all != 0).
                    eprintln!("Error: {e:?}");
                    std::process::exit(kind.exit_code());
                }
            }
        }
        _ = interrupted => {
            model::emit_progress_event(&model::ProgressEvent::Error {
                kind: model::ProgressErrorKind::Interrupted,
                message: "interrupted by SIGINT".to_string(),
            });
            eprintln!("Interrupted by Ctrl-C");
            std::process::exit(model::ProgressErrorKind::Interrupted.exit_code());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fp32_manifest_is_usable_by_engine_preflight() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_str().unwrap();
        let variant = ModelVariant::E2eRnnt;
        for name in variant.download_files() {
            std::fs::write(tmp.path().join(name), b"model fixture").unwrap();
        }
        ensure_fp32_manifest(dir, variant).unwrap();
        let path = tmp.path().join(model::MANIFEST_FILE);
        let external = format!(
            "# Prepared externally\n{}",
            std::fs::read_to_string(&path).unwrap()
        );
        std::fs::write(&path, &external).unwrap();
        ensure_fp32_manifest(dir, variant).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), external);
        assert_eq!(
            model::ensure_model_variant(Some(variant), dir)
                .await
                .unwrap(),
            variant
        );
    }

    #[test]
    fn test_fp32_manifest_preserves_existing_custom_pack() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(model::MANIFEST_FILE);
        let existing = "architecture = \"custom\"\n";
        std::fs::write(&path, existing).unwrap();
        assert!(ensure_fp32_manifest(tmp.path().to_str().unwrap(), ModelVariant::E2eRnnt).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), existing);
    }

    #[test]
    fn test_fp32_manifest_rejects_unsupported_head() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(ensure_fp32_manifest(tmp.path().to_str().unwrap(), ModelVariant::Rnnt).is_err());
        assert!(!tmp.path().join(model::MANIFEST_FILE).exists());
    }

    #[test]
    fn test_run_quantize_explicit_head_overrides_directory_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let rnnt = tmp.path().join("v3_rnnt_encoder_int8.onnx");
        std::fs::write(&rnnt, b"existing rnnt").unwrap();
        let error = run_quantize(
            tmp.path().display().to_string(),
            true,
            false,
            Some(ModelVariant::E2eRnnt),
        )
        .unwrap_err();
        assert!(error.to_string().contains("v3_e2e_rnnt_encoder.onnx"));
        assert_eq!(std::fs::read(rnnt).unwrap(), b"existing rnnt");
    }

    #[test]
    fn test_run_quantize_refuses_existing_int8_without_force() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        run_quantize(tmp.path().display().to_string(), false, false, None)
            .expect_err("existing INT8 needs explicit replacement");
        assert!(!tmp.path().join("v3_rnnt_encoder.onnx").exists());
    }

    #[test]
    fn test_run_quantize_legacy_recipe_refuses_existing_int8_without_force() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let int8 = tmp.path().join("v3_rnnt_encoder_int8.onnx");
        std::fs::write(&int8, b"int8").unwrap();
        let err = run_quantize(tmp.path().display().to_string(), false, true, None)
            .expect_err("legacy recipe must not succeed over an existing INT8 file");
        let msg = format!("{err:#}");
        assert!(msg.contains("--force"), "unexpected error: {msg}");
        assert_eq!(std::fs::read(&int8).unwrap(), b"int8");
    }

    #[test]
    fn test_run_quantize_errors_when_fp32_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = run_quantize(tmp.path().display().to_string(), true, false, None)
            .expect_err("force still needs FP32");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("FP32 encoder not found"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_run_cache_gc_empty_dir_is_ok() {
        let tmp = tempfile::tempdir().expect("tempdir");
        run_cache_gc(tmp.path().display().to_string(), true, true, None).expect("empty dry-run");
        run_cache_gc(tmp.path().display().to_string(), false, false, None).expect("empty prune");
    }

    #[test]
    fn test_run_cache_gc_prunes_explicit_optimized_cache_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().join("models");
        // rnnt INT8 stub so the prune has a keep-name.
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        let cache_dir = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let keep = cache_dir.join(format!(
            "cb1525bced78da2c03c42fe15bf15663b584566ef6244ff91d892caa011fec1e-{}_optimized.ort",
            "a".repeat(64)
        ));
        let zombie = cache_dir.join("v3_e2e_rnnt_encoder_int8_optimized.ort");
        std::fs::write(&keep, b"keep").unwrap();
        std::fs::write(&zombie, b"zombie").unwrap();

        run_cache_gc(
            model_dir.display().to_string(),
            false,
            false,
            Some(cache_dir.display().to_string()),
        )
        .expect("prune explicit cache dir");
        assert!(keep.exists());
        assert!(!zombie.exists());
    }
}
