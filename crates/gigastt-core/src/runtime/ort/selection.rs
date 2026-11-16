//! Operator choice of ORT execution provider, and what this build can bind.
//!
//! `auto` may fall back to CPU. An exact provider must be compiled into the
//! binary; the load path then refuses to rebuild the pool on CPU.

use crate::model::ModelVariant;

/// What the operator asked for. `Auto` is the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProviderChoice {
    Auto,
    Cpu,
    Cuda,
}

impl ExecutionProviderChoice {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown execution provider '{other}' (expected auto, cpu, cuda)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    pub fn allows_cpu_fallback(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Provider this binary will try to bind before any runtime fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundProvider {
    Cpu,
    Cuda,
}

impl BoundProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

pub fn compiled_ort_default() -> BoundProvider {
    if cfg!(feature = "cuda") {
        BoundProvider::Cuda
    } else {
        BoundProvider::Cpu
    }
}

pub fn auto_backend_name(_variant: ModelVariant) -> &'static str {
    compiled_ort_default().as_str()
}

/// Exact provider, or an error when this build cannot load it.
pub fn resolve_ort_provider(choice: ExecutionProviderChoice) -> Result<BoundProvider, String> {
    match choice {
        ExecutionProviderChoice::Auto => Ok(compiled_ort_default()),
        ExecutionProviderChoice::Cpu => Ok(BoundProvider::Cpu),

        ExecutionProviderChoice::Cuda => {
            if cfg!(feature = "cuda") {
                Ok(BoundProvider::Cuda)
            } else {
                Err(
                    "execution provider 'cuda' was requested but this build does not include it \
                     (rebuild with --features cuda, or use auto/cpu)"
                        .into(),
                )
            }
        }
    }
}

/// After the primary provider fails to load or to pass warmup: `auto` binds
/// CPU; an exact request fails.
pub fn bind_after_primary_failure(choice: ExecutionProviderChoice) -> Result<&'static str, ()> {
    if choice.allows_cpu_fallback() {
        Ok("cpu")
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_execution_provider_names() {
        assert_eq!(
            ExecutionProviderChoice::parse("auto").unwrap(),
            ExecutionProviderChoice::Auto
        );
        assert_eq!(
            ExecutionProviderChoice::parse("CPU").unwrap(),
            ExecutionProviderChoice::Cpu
        );
        assert_eq!(
            ExecutionProviderChoice::parse("cuda").unwrap(),
            ExecutionProviderChoice::Cuda
        );
        for name in ["coreml", "ane", "candle", "nnapi", "vulkan"] {
            assert!(ExecutionProviderChoice::parse(name).is_err());
        }
    }

    #[test]
    fn test_auto_may_fall_back_and_exact_may_not() {
        assert_eq!(
            bind_after_primary_failure(ExecutionProviderChoice::Auto),
            Ok("cpu")
        );
        assert!(bind_after_primary_failure(ExecutionProviderChoice::Cuda).is_err());
        assert!(bind_after_primary_failure(ExecutionProviderChoice::Cpu).is_err());
    }

    #[test]
    fn test_exact_cpu_binds_cpu() {
        assert_eq!(
            resolve_ort_provider(ExecutionProviderChoice::Cpu).unwrap(),
            BoundProvider::Cpu
        );
        assert_eq!(BoundProvider::Cpu.as_str(), "cpu");
    }

    #[test]
    fn test_exact_unavailable_provider_fails() {
        let cuda = resolve_ort_provider(ExecutionProviderChoice::Cuda);
        if cfg!(feature = "cuda") {
            assert_eq!(cuda.unwrap(), BoundProvider::Cuda);
        } else {
            let msg = cuda.unwrap_err();
            assert!(msg.contains("cuda"), "{msg}");
        }
    }
}
