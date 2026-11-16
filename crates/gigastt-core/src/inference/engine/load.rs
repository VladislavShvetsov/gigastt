//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Load ONNX models from the given directory and create an inference engine.
    ///
    /// Creates a pool of `DEFAULT_POOL_SIZE` session triplets for concurrent inference.
    /// The recognition head ([`ModelVariant`]) is auto-detected from the encoder
    /// file present on disk: `v3_rnnt_encoder.onnx` (or `_int8.onnx`) selects the
    /// plain rnnt head, else `v3_e2e_rnnt_encoder.onnx` selects e2e_rnnt. The
    /// matching decoder, joiner, and vocab files must also be present.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, GigasttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size. Requires the *full* pool to
    /// load (every triplet); use [`Engine::load_with_pool_size_min`] to boot on
    /// a partial pool.
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, GigasttError> {
        Self::load_with_pool_size_min(model_dir, pool_size, pool_size)
    }

    /// Load ONNX models with a custom pool size, tolerating a partial pool down
    /// to `min_size` triplets when the rest fail to load. Boots a degraded pool
    /// with a warning when `min_size <= loaded < pool_size`; errors only when
    /// fewer than `min_size` triplets load. `min_size` is clamped to
    /// `1..=pool_size`.
    pub fn load_with_pool_size_min(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
    ) -> Result<Self, GigasttError> {
        // No batch/stream split: the whole pool is interactive.
        Self::load_with_pools(model_dir, pool_size, min_size, 0)
    }

    /// Load ONNX models splitting the pool into an interactive pool (WebSocket +
    /// SSE) and a dedicated batch pool of `batch_pool_size` triplets for REST
    /// file transcription, so a long batch job can't starve real-time
    /// streaming. `batch_pool_size == 0` disables the split (REST shares the
    /// interactive pool); it is clamped to leave at least one interactive
    /// triplet. Partial-load tolerance follows `min_size` as in
    /// [`Engine::load_with_pool_size_min`].
    pub fn load_with_pools(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pools_threads(model_dir, pool_size, min_size, batch_pool_size, 1)
    }

    pub fn load_with_pools_threads(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pools_threads_variant(
            model_dir,
            None,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
        )
    }

    /// Like [`Engine::load_with_pools_threads`], but lets the caller force which
    /// recognition head to load instead of auto-detecting it.
    ///
    /// When `variant` is `Some(v)`, head `v` is loaded (and the load fails with a
    /// clear `ModelLoad` error if `v`'s files aren't in `model_dir`). When it is
    /// `None`, the on-disk layout is auto-detected with `rnnt` precedence, exactly
    /// as [`Engine::load_with_pools_threads`]. This is the entry point that makes
    /// `--model-variant` effective when a directory holds more than one head.
    pub fn load_with_pools_threads_variant(
        model_dir: &str,
        variant: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pools_threads_variant_cache(
            model_dir,
            variant,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
            None,
        )
    }

    /// Like [`Engine::load_with_pools_threads_variant`], with an optional
    /// override for the CPU encoder's ORT optimized-graph cache directory.
    /// `None` keeps the default `<model_dir>/optimized_cache`; `Some(dir)`
    /// relocates the cache (e.g. a writable cache directory when the model
    /// dir is read-only). Only used by the CPU ort backend.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_pools_threads_variant_cache(
        model_dir: &str,
        variant: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
        optimized_cache_dir: Option<PathBuf>,
    ) -> Result<Self, GigasttError> {
        Self::load_with_execution_provider(
            model_dir,
            variant,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
            optimized_cache_dir,
            crate::runtime::ExecutionProviderChoice::Auto,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with_execution_provider(
        model_dir: &str,
        variant: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
        optimized_cache_dir: Option<PathBuf>,
        execution_provider: crate::runtime::ExecutionProviderChoice,
    ) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        // Resolve the head once, up front: an explicit `variant` wins, else
        // manifest.toml architecture, else detect from disk (rnnt precedence).
        // Resolving here (not just inside `load_with_factory`) keeps the RAM cap
        // sized to the head that will actually load.
        let variant = resolve_variant_required(variant, dir)?;
        // Bound the idle footprint: each pooled triplet deserializes its own
        // encoder copy, so a large `--pool-size` on a small host can OOM at
        // load. Clamp by available RAM (logs when it clamps); a no-op on hosts
        // with ample memory.
        let encoder_bytes = std::fs::metadata(encoder_model_path(dir, variant))
            .map(|m| m.len())
            .unwrap_or(0);
        let pool_size =
            Self::cap_pool_size_for_ram(pool_size, encoder_bytes, sizing::effective_ram_bytes());
        // Don't let `pool_size * encoder_intra_threads` oversubscribe the CPU
        // (no-op when the default `1` is requested).
        let logical_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let encoder_intra_threads =
            Self::clamp_encoder_intra_threads(pool_size, encoder_intra_threads, logical_cpus);

        let provider_error = |message: String| GigasttError::ModelLoad {
            path: model_dir.to_string(),
            source: Some(message.into()),
        };
        let allow_cpu_fallback =
            crate::runtime::bind_after_primary_failure(execution_provider).is_ok();
        let (factory, backend_name) = match execution_provider {
            crate::runtime::ExecutionProviderChoice::Auto => (
                production_factory_variant_with_cache(
                    dir,
                    Some(variant),
                    optimized_cache_dir.clone(),
                ),
                crate::runtime::auto_backend_name(variant),
            ),
            exact => {
                let bound = crate::runtime::resolve_ort_provider(exact).map_err(provider_error)?;
                (
                    crate::runtime::exact_ort_factory(bound, dir, optimized_cache_dir.clone()),
                    bound.as_str(),
                )
            }
        };
        Self::load_with_factory(
            dir,
            Some(variant),
            pool_size,
            min_size,
            batch_pool_size,
            factory,
            encoder_intra_threads,
            allow_cpu_fallback,
            backend_name,
        )
    }

    /// Assemble an engine from already-loaded sessions, tokenizer, and variant.
    ///
    /// Optional attachments (punctuator, ITN, biaser, VAD) stay off — callers
    /// chain [`Engine::with_punctuator`] / [`Engine::with_itn`] / etc.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_loaded_parts(
        pool: SessionPool,
        batch_pool: Option<SessionPool>,
        tokenizer: Tokenizer,
        variant: ModelVariant,
        int8: bool,
        ane_encoder: bool,
        execution_provider: impl Into<String>,
        #[cfg(feature = "diarization")] speaker_encoder: Option<LazySpeakerEncoder>,
    ) -> Self {
        Self {
            pool,
            batch_pool,
            tokenizer,
            features: if int8 {
                FeatureExtractor::new()
            } else {
                FeatureExtractor::for_variant(variant)
            },
            variant,
            punctuator: None,
            itn: false,
            biaser: None,
            vad: None,
            vad_config: crate::vad::VadConfig::default(),
            endpoint_mode: EndpointMode::Auto,
            endpoint_silence: crate::inference::EndpointSilence::default(),
            stream_max_window_samples: STREAM_MAX_WINDOW_SAMPLES,
            // Default on: stable-prefix commits cut long-phrase streaming WER
            // loss at slide boundaries (10.6% → 2.1% corpus WER on 10 labelled
            // Golos fixtures at the default 2.5 s window) with no measurable
            // TTFP or encoder-cost change.
            stream_stable_prefix: true,
            int8,
            ane_encoder,
            execution_provider: execution_provider.into(),
            file_window_concurrency: 1,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        }
    }

    /// Package-private factory-based loader. Used by production code paths and
    /// by tests that inject a [`crate::runtime::factory::RuntimeFactory`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_with_factory(
        model_dir: &Path,
        variant_override: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        factory: Box<dyn crate::runtime::factory::RuntimeFactory>,
        encoder_intra_threads: usize,
        allow_cpu_fallback: bool,
        execution_provider: &'static str,
    ) -> Result<Self, GigasttError> {
        // Honor an explicit variant (e.g. from `--model-variant`) when the caller
        // resolved one; otherwise prefer `manifest.toml`, else auto-detect from
        // the on-disk layout (rnnt precedence). Passing the override through is
        // what makes `--model-variant` effective when a directory holds more than
        // one head — without it the engine always re-detects and silently loads
        // the highest-precedence head.
        let variant = resolve_variant_required(variant_override, model_dir)?;
        let pool_size = pool_size.max(1);
        let min_size = min_size.clamp(1, pool_size);

        let runtime = factory.create(encoder_intra_threads)?;
        let model_load = |e: anyhow::Error| GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: Some(e.into()),
        };

        let mut files = ResolvedModelFiles::resolve(model_dir, variant).map_err(model_load)?;
        if factory.verify_on_disk_checksums() {
            files.verify_pinned_checksums(variant)?;
        }

        tracing::info!("Detected model variant: {variant:?}");
        let is_int8 = files.using_int8;
        {
            if files.using_int8 {
                tracing::info!("Using INT8 quantized encoder");
            } else {
                tracing::info!("Using explicitly configured FP32 encoder");
            }
            tracing::info!(
                "Loading ONNX models from {} (pool_size={pool_size})...",
                model_dir.display()
            );
        }

        #[cfg(feature = "cuda")]
        tracing::info!("Using CUDA execution provider (falls back to CPU if unavailable)");
        #[cfg(not(feature = "cuda"))]
        tracing::info!("Using CPU execution provider");

        let fell_back = false;

        let _ = allow_cpu_fallback;

        let triplets = load_triplets_runtime(&*runtime, &files, variant, pool_size, min_size)
            .map_err(model_load)?;

        let tokenizer = Tokenizer::load(&files.vocab).map_err(model_load)?;

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        // Probe only — do not open the WeSpeaker encoder session at boot. The
        // encoder is loaded on the first diarization request.
        #[cfg(feature = "diarization")]
        let speaker_encoder = diarization::probe_speaker_encoder(model_dir);

        let ane_encoder = triplets.first().is_some_and(|t| t.encoder.is_ane_encoder());
        let (pool, batch_pool) = Self::split_triplets(triplets, batch_pool_size);
        let engine = Self::from_loaded_parts(
            pool,
            batch_pool,
            tokenizer,
            variant,
            is_int8,
            ane_encoder,
            if fell_back { "cpu" } else { execution_provider },
            #[cfg(feature = "diarization")]
            speaker_encoder,
        );

        tracing::info!(
            execution_provider = engine.execution_provider(),
            "execution provider bound"
        );
        Ok(engine)
    }

    /// Split loaded triplets into an interactive pool and an optional batch
    /// pool of `batch_pool_size` triplets. Always leaves at least one triplet
    /// for the interactive pool; `batch_pool_size == 0` (or a pool too small to
    /// split) yields no batch pool.
    pub(crate) fn split_triplets(
        triplets: Vec<SessionTriplet>,
        batch_pool_size: usize,
    ) -> (SessionPool, Option<SessionPool>) {
        Self::split_pool(triplets, batch_pool_size)
    }

    /// Generic pool split underlying [`Engine::split_triplets`]: partition
    /// `items` into an interactive pool and an optional batch pool of
    /// `batch_pool_size` items, always leaving at least one item interactive.
    /// `batch_pool_size == 0` (or too few items to split) yields no batch pool.
    /// Generic over the item type so the routing can be unit-tested with a
    /// synthetic `Pool<u32>` instead of model-backed `SessionTriplet`s.
    pub(crate) fn split_pool<T: Send>(
        items: Vec<T>,
        batch_pool_size: usize,
    ) -> (Pool<T>, Option<Pool<T>>) {
        let (interactive, batch) = sizing::split_pool_items(items, batch_pool_size);
        (Pool::new(interactive), batch.map(Pool::new))
    }

    /// See [`sizing::cap_pool_size_for_ram`].
    pub(crate) fn cap_pool_size_for_ram(
        requested: usize,
        encoder_bytes: u64,
        total_ram: u64,
    ) -> usize {
        sizing::cap_pool_size_for_ram(requested, encoder_bytes, total_ram)
    }

    /// See [`sizing::clamp_encoder_intra_threads`].
    pub(crate) fn clamp_encoder_intra_threads(
        pool_size: usize,
        requested: usize,
        logical_cpus: usize,
    ) -> usize {
        sizing::clamp_encoder_intra_threads(pool_size, requested, logical_cpus)
    }

    pub(crate) fn warmup_one(&self) -> Result<(), GigasttError> {
        self.warmup_one_on(&self.pool)
    }

    /// Warm a single triplet from a specific pool with ~1 s of silence.
    pub(crate) fn warmup_one_on(&self, pool: &SessionPool) -> Result<(), GigasttError> {
        let silence = vec![0.0f32; 16000]; // 1 s at 16 kHz
        let mut guard = pool
            .checkout_blocking()
            .map_err(|e| GigasttError::Inference {
                source: Box::new(e),
            })?;
        self.transcribe_samples(&silence, &mut guard)?;
        Ok(())
    }

    pub fn warmup(&self) -> Result<(), GigasttError> {
        for _ in 0..self.pool.total() {
            self.warmup_one()?;
        }
        if let Some(ref batch) = self.batch_pool {
            for _ in 0..batch.total() {
                self.warmup_one_on(batch)?;
            }
        }
        Ok(())
    }

    /// The pool REST file transcription should use: the dedicated batch pool
    /// when one was split off, otherwise the interactive pool.
    pub fn pool_for_batch(&self) -> &SessionPool {
        self.batch_pool.as_ref().unwrap_or(&self.pool)
    }

    /// Close both the interactive and batch pools so every waiter wakes with
    /// `PoolError::Closed` during graceful shutdown.
    pub fn close_pools(&self) {
        self.pool.close();
        if let Some(ref batch) = self.batch_pool {
            batch.close();
        }
    }
}
