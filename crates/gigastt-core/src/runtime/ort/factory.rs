#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
};

use super::session::OrtRuntime;

#[derive(Clone, Copy)]
pub enum OrtExecutionProvider {
    Cpu,

    #[cfg(feature = "cuda")]
    Cuda,
}

impl OrtExecutionProvider {
    pub(crate) fn execution_providers(
        self,
        model_path: &Path,
        secondary_cpu: bool,
    ) -> Vec<ort::ep::ExecutionProviderDispatch> {
        let _ = model_path;
        #[cfg(not(feature = "cuda"))]
        let _ = secondary_cpu;
        match self {
            Self::Cpu => vec![ort::ep::CPU::default().build()],

            #[cfg(feature = "cuda")]
            Self::Cuda => {
                // `auto` keeps CPU as a second provider so a missing GPU still
                // loads. An exact `cuda` request omits it: session creation
                // fails instead of silently running on CPU.
                let mut eps = vec![ort::ep::CUDA::default().build()];
                if secondary_cpu {
                    eps.push(ort::ep::CPU::default().build());
                }
                eps
            }
        }
    }

    /// Whether this provider is the plain CPU execution provider.
    pub(crate) fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

/// Factory that creates an `ort` runtime configured for a specific provider.
pub struct OrtFactory {
    provider: OrtExecutionProvider,
    secondary_cpu: bool,
    prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
    optimized_cache_dir: Option<PathBuf>,
}

impl OrtFactory {
    fn with_provider(provider: OrtExecutionProvider) -> Self {
        Self {
            provider,
            secondary_cpu: true,
            prepacked: None,
            optimized_cache_dir: None,
        }
    }

    /// CUDA with no CPU provider. Session load fails when the GPU is absent.
    #[cfg(feature = "cuda")]
    pub fn cuda_exact() -> Self {
        let mut factory = Self::cuda();
        factory.secondary_cpu = false;
        factory
    }

    pub fn cpu() -> Self {
        Self::with_provider(OrtExecutionProvider::Cpu)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Self {
        Self::with_provider(OrtExecutionProvider::Cuda)
    }

    pub fn with_prepacked_weights(
        mut self,
        prepacked: Arc<ort::session::builder::PrepackedWeights>,
    ) -> Self {
        self.prepacked = Some(prepacked);
        self
    }

    pub fn with_optimized_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.optimized_cache_dir = Some(dir.into());
        self
    }
}

// ort permits environment creation only once per process. Retain the default
// across independent engines and admin reloads; runtimes and sessions clone it.
static ORT_ENVIRONMENT: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();

pub(crate) fn shared_environment() -> Result<ort::environment::Environment, RuntimeError> {
    ORT_ENVIRONMENT
        .get_or_init(|| {
            let environment = ort::init()
                .with_name("gigastt")
                .build()
                .map_err(|error| error.to_string())?;
            tracing::info!(runtime_info = ort::info(), "ONNX Runtime initialized");
            Ok(environment)
        })
        .clone()
        .map_err(|message| RuntimeError::LoadFailed {
            path: std::path::PathBuf::new(),
            message: format!("runtime initialization failed: {message}"),
        })
}

impl RuntimeFactory for OrtFactory {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        let environment = shared_environment()?;
        Ok(Box::new(OrtRuntime::new(
            environment,
            intra_threads,
            self.provider,
            self.secondary_cpu,
            self.prepacked.clone(),
            self.optimized_cache_dir.clone(),
        )))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(OrtFactory::cpu())
    }
}

pub fn default_factory() -> Box<dyn RuntimeFactory> {
    {
        #[cfg(feature = "cuda")]
        {
            Box::new(OrtFactory::cuda())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Box::new(OrtFactory::cpu())
        }
    }
}

/// Returns a CPU-only `ort` factory for auxiliary models.
pub fn cpu_factory() -> Box<dyn RuntimeFactory> {
    Box::new(OrtFactory::cpu())
}

/// Speaker pools need threaded CPU inference without idle workers competing
/// with another slot or the recognition encoder between embedding calls.
#[cfg(feature = "diarization")]
pub(crate) fn speaker_runtime(intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
    let environment = shared_environment()?;
    Ok(Box::new(
        OrtRuntime::new(
            environment,
            intra_threads,
            OrtExecutionProvider::Cpu,
            true,
            None,
            None,
        )
        .without_spinning(),
    ))
}

/// Returns a production `ort` factory that preserves the provider selection and
/// disk-cache layout used by the engine before the runtime abstraction.
///
/// Public, stable 1-arg form: selects the backend from the variant detected on
/// disk. The engine calls the crate-internal `production_factory_variant_with_cache`
/// instead, passing the head it has already resolved so an explicit
/// `--model-variant` is honored.
pub fn production_factory(model_dir: &Path) -> Box<dyn RuntimeFactory> {
    production_factory_variant(
        model_dir,
        crate::model::ModelVariant::detect_in_dir(model_dir),
    )
}

/// Build the ort factory for an exact provider. CPU keeps the optimized-graph
/// cache. Exact CUDA does not register a CPU provider beside it.
pub(crate) fn exact_ort_factory(
    provider: super::selection::BoundProvider,
    model_dir: &Path,
    optimized_cache_dir: Option<PathBuf>,
) -> Box<dyn RuntimeFactory> {
    match provider {
        super::selection::BoundProvider::Cpu => {
            let prepacked = std::sync::Arc::new(ort::session::builder::PrepackedWeights::new());
            let cache = optimized_cache_dir.unwrap_or_else(|| model_dir.join("optimized_cache"));
            Box::new(
                OrtFactory::cpu()
                    .with_optimized_cache_dir(cache)
                    .with_prepacked_weights(prepacked),
            )
        }

        super::selection::BoundProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let _ = (model_dir, optimized_cache_dir);
                Box::new(OrtFactory::cuda_exact())
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, optimized_cache_dir);
                unreachable!("cuda is rejected before this factory is built")
            }
        }
    }
}

pub(crate) fn production_factory_variant(
    model_dir: &Path,
    variant: Option<crate::model::ModelVariant>,
) -> Box<dyn RuntimeFactory> {
    production_factory_variant_with_cache(model_dir, variant, None)
}

/// Like [`production_factory_variant`], with an optional override for the CPU
/// encoder's ORT optimized-graph cache directory. `None` keeps the default
/// `<model_dir>/optimized_cache`; `Some(dir)` relocates it (e.g. a writable
/// `CacheDirectory` when the model dir is read-only under systemd
/// `ProtectSystem=strict`).
pub(crate) fn production_factory_variant_with_cache(
    model_dir: &Path,
    variant: Option<crate::model::ModelVariant>,
    optimized_cache_dir: Option<PathBuf>,
) -> Box<dyn RuntimeFactory> {
    let _ = variant;

    #[cfg(feature = "cuda")]
    let factory = OrtFactory::cuda();
    #[cfg(not(feature = "cuda"))]
    let factory = {
        // Shared PrepackedWeights across every session this factory creates.
        // ORT still materializes per-session initializers for most graphs; the
        // container shares prepacked kernel buffers when the EP supports it.
        // Enabled so an extra pool slot can share prepacked kernel buffers.
        // Remeasure the pool-1→2 RSS after ORT upgrades. Safe no-op if unused.
        let prepacked = std::sync::Arc::new(ort::session::builder::PrepackedWeights::new());
        OrtFactory::cpu()
            .with_optimized_cache_dir(
                optimized_cache_dir.unwrap_or_else(|| model_dir.join("optimized_cache")),
            )
            .with_prepacked_weights(prepacked)
    };
    #[cfg(feature = "cuda")]
    let _ = (model_dir, optimized_cache_dir);
    Box::new(factory)
}

#[cfg(test)]
mod tests {
    use super::*;

    // `PrepackedWeights::new` calls `OrtGetApiBase`. Miri aborts on that
    // foreign function instead of failing the test.
    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cpu_factory_can_attach_prepacked_weights() {
        let pw = std::sync::Arc::new(ort::session::builder::PrepackedWeights::new());
        let f = OrtFactory::cpu().with_prepacked_weights(pw);
        // create() must succeed without a model path (runtime shell only).
        let rt = f.create(1).expect("cpu runtime with prepacked");
        drop(rt);
    }
}
