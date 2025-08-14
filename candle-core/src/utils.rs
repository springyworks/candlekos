//! Useful functions for checking features.
use std::str::FromStr;

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx2")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}

/// Lightweight debug macro for FFT development.
/// Usage: fft_debug!("message: {}", value);
/// Activated if compiled with `--features fft-debug` OR env var CANDLE_FFT_DEBUG=1/true at runtime.
#[macro_export]
macro_rules! fft_debug {
    ($($arg:tt)*) => {{
        #[allow(unused)]
        {
            let enabled = cfg!(feature = "fft-debug") || std::env::var("CANDLE_FFT_DEBUG").map(|v| v=="1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
            if enabled { eprintln!("[fft-debug] {}", format!($($arg)*)); }
        }
    }};
}
