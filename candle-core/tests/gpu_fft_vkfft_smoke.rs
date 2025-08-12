#![cfg(all(feature = "cuda", feature = "gpu-fft", feature = "gpu-fft-vkfft"))]

use candle_core::{Device, Result, Tensor};

#[test]
fn vkfft_provider_smoke_returns_placeholder_error() -> Result<()> {
    // Create a small CUDA tensor and try an FFT call. The VkFFT provider scaffold
    // should return a clear placeholder error. This verifies provider wiring only.
    let dev = Device::new_cuda(0)?;
    let t = Tensor::randn(0f32, 1f32, (8,), &dev)?;
    let err = t.fftn([0usize], false, false).unwrap_err();
    let msg = format!("{}", err);
    let ok = msg.contains("VkFFT provider selected, but FFI is not yet wired")
        || msg.contains("VkFFT FFI enabled (version ");
    assert!(ok, "unexpected error: {msg}");
    Ok(())
}
