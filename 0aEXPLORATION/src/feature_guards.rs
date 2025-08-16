//! Compile-time feature invariants and helper cfg gates.
//!
//! This module enforces relationships between cargo features so that
//! invalid combinations fail fast with a clear error message.
//!
//! Invariants:
//! 1. `cudnn` implies `cuda`.
//! 2. `fft` can be used standalone on CPU, or with `cuda` when GPU FFT is supported.
//!    Currently no strict coupling enforced. If in future certain FFT kernels require
//!    CUDA (e.g., a `gpu-fft` sub-feature) add a compile_error! invariant here.
//!
//! Add new invariants here as features grow.

#[cfg(all(feature = "cudnn", not(feature = "cuda")))]
compile_error!(
    "Feature 'cudnn' requires feature 'cuda'. Enable with --features=cuda,cudnn (or remove cudnn)."
);

// Example (currently permissive) placeholder for a future invariant:
// #[cfg(all(feature = "gpu-only-future", not(feature = "cuda")))]
// compile_error!("'gpu-only-future' requires 'cuda'.");

// Potential future: enforce that cudnn implies cuda+fft if FFT acceleration depends on cudnn.
// #[cfg(all(feature = "cudnn", feature = "fft", not(feature = "cuda")))]
// compile_error!("Internal logic: cudnn+fft path expects cuda.");

/// Whether GPU acceleration (CUDA) is compiled in.
#[allow(dead_code)]
pub const HAS_CUDA: bool = cfg!(feature = "cuda");
/// Whether cuDNN support is compiled in.
#[allow(dead_code)]
pub const HAS_CUDNN: bool = cfg!(feature = "cudnn");
/// Whether FFT support is compiled in.
#[allow(dead_code)]
pub const HAS_FFT: bool = cfg!(feature = "fft");
