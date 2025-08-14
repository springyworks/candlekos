#![cfg(all(feature = "cuda", feature = "fft", feature = "gpu-fft"))]
// Minimal GPU FFT forward + inverse roundtrip to ensure baseline provider path compiles & runs.
// This is intentionally tiny: it should catch linkage / provider selection issues early without
// duplicating the heavier vkFFT suites.

use candle_core::{Device, Result, Tensor};
mod fft_test_utils; // shared helpers
use fft_test_utils::{assert_approx_scaled, expected_range, FFT_EPS_REAL};

#[test]
fn gpu_fft_forward_inverse_roundtrip() -> Result<()> {
    // Gracefully skip if CUDA device or runtime not available.
    let dev = match Device::new_cuda(0) { Ok(d) => d, Err(_) => return Ok(()) };
    // Simple real range tensor.
    let t = Tensor::arange(0f32, 16f32, &dev)?; // length 16
    let freq = t.rfft(0, false)?; // real -> complex (packed real/imag)
    assert!(freq.dims()[0] >= (16 / 2 + 1) * 2, "unexpected packed spectrum size");
    let inv = freq.irfft(0, false)?; // inverse (un-normalized)
    let v = inv.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    assert_eq!(v.len(), 16);
    assert_approx_scaled(&v, &expected_range(16), FFT_EPS_REAL);
    Ok(())
}
