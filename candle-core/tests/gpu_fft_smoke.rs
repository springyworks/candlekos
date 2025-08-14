#![cfg(all(feature = "cuda", feature = "fft", feature = "gpu-fft"))]
// Minimal GPU FFT forward + inverse roundtrip to ensure baseline provider path compiles & runs.
// This is intentionally tiny: it should catch linkage / provider selection issues early without
// duplicating the heavier vkFFT suites.

use candle_core::{Device, Result, Tensor};

#[test]
fn gpu_fft_forward_inverse_roundtrip() -> Result<()> {
    // Gracefully skip if CUDA device or runtime not available.
    let dev = match Device::new_cuda(0) { Ok(d) => d, Err(_) => return Ok(()) };
    // Simple real range tensor.
    let t = Tensor::arange(0f32, 16f32, &dev)?; // length 16
    let freq = t.rfft(0, false)?; // real -> complex (packed real/imag)
    assert!(freq.dims()[0] >= (16 / 2 + 1) * 2, "unexpected packed spectrum size");
    let inv = freq.irfft(0, false)?; // inverse (un-normalized)
    let inv_cpu = inv.to_device(&Device::Cpu)?;
    let v = inv_cpu.to_vec1::<f32>()?;
    assert_eq!(v.len(), 16);
    // Expected (unscaled) inverse of forward real FFT should recover original values (within fp error)
    // Some providers may apply implicit scaling; detect and normalize if needed.
    let orig: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // Detect uniform scale: v[i]/orig[i] (skip i=0) should be ~constant if scaled.
    let mut scale = None;
    for i in 1..v.len() { if orig[i] != 0.0 { scale = Some(v[i]/orig[i]); break; } }
    if let Some(s) = scale { if (s - 1.0).abs() > 1e-3 { // normalize if clearly scaled
        for (vi, oi) in v.iter().zip(orig.iter()) { assert!((vi / s - oi).abs() < 5e-3, "roundtrip mismatch after scale normalization: vi={vi} oi={oi} scale={s}"); }
    } else {
        for (vi, oi) in v.iter().zip(orig.iter()) { assert!((vi - oi).abs() < 5e-3, "roundtrip mismatch: vi={vi} oi={oi}"); }
    }}
    Ok(())
}
