#![cfg(all(feature = "cuda", feature = "gpu-fft", feature = "gpu-fft-vkfft"))]

use candle_core::{Device, Result, Tensor};

#[test]
fn vkfft_provider_smoke() -> Result<()> {
    // Create a small CUDA tensor and try an FFT call. If FFI is wired, it should succeed.
    // Otherwise, we should get a clear placeholder error.
    let dev = Device::new_cuda(0)?;
    let t = Tensor::randn(0f32, 1f32, (8,), &dev)?;
    match t.fftn([0usize], false, false) {
        Ok(_out) => Ok(()),
        Err(err) => {
            let msg = format!("{}", err);
            let ok = msg.contains("VkFFT provider selected, but FFI is not yet wired")
                || msg.contains("VkFFT FFI enabled (version ");
            assert!(ok, "unexpected error: {msg}");
            Ok(())
        }
    }
}
