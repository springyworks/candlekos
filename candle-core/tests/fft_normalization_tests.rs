#![cfg(feature = "fft")]
// Tests for normalization semantics of forward/inverse FFT.

use candle_core::{Device, Result, Tensor};
mod fft_test_utils; // reuse helpers
use fft_test_utils::{FFT_EPS_REAL, assert_approx_scaled, expected_range};

#[test]
fn fft_normalized_roundtrip_matches() -> Result<()> {
    let dev = Device::Cpu;
    let n = 32;
    let orig = Tensor::arange(0f32, n as f32, &dev)?;
    // Forward normalized + inverse normalized should preserve scale within epsilon.
    let freq = orig.rfft(0, true)?; // real -> complex normalized
    let back = freq.irfft(0, true)?; // inverse normalized
    let back_vec = back.to_vec1::<f32>()?;
    assert_approx_scaled(&back_vec, &expected_range(n), FFT_EPS_REAL);
    Ok(())
}

#[test]
fn fft_mixed_normalization_has_scale() -> Result<()> {
    let dev = Device::Cpu;
    let n = 32;
    let orig = Tensor::arange(0f32, n as f32, &dev)?;
    // Forward normalized=false, inverse normalized=false will often introduce n scaling
    let freq = orig.rfft(0, false)?;
    let back = freq.irfft(0, false)?;
    let v = back.to_vec1::<f32>()?;
    // After dividing by n we should match.
    let scaled: Vec<f32> = v.iter().map(|x| *x / n as f32).collect();
    assert_approx_scaled(&scaled, &expected_range(n), FFT_EPS_REAL * 2.0); // allow small slack
    Ok(())
}
