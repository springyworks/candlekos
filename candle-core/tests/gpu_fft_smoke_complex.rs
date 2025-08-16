#![cfg(all(feature = "cuda", feature = "fft", feature = "gpu-fft"))]
// Complex-to-complex GPU FFT smoke test: creates an interleaved complex tensor,
// performs a forward + inverse FFT roundtrip along one dimension and validates
// reconstruction within tolerance (considering possible provider scaling).

use candle_core::{Device, Result, Tensor};
mod fft_test_utils; // shared helpers
use fft_test_utils::{
    FFT_EPS_COMPLEX, assert_approx_scaled, expected_sin_cos, split_interleaved_complex,
};

#[test]
fn gpu_fft_c2c_roundtrip() -> Result<()> {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let n = 32usize;
    // Interleaved complex layout: [re0, im0, re1, im1, ...]
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32;
        data.push(t.sin());
        data.push(t.cos());
    }
    let complex = Tensor::from_vec(data, &[n * 2], &dev)?; // Single axis length n (interleaved)

    // Forward complex FFT (real_input=false)
    let freq = complex.fft(0, false, false)?;
    // Inverse
    let time = freq.ifft(0, false)?;
    let host = time.to_device(&Device::Cpu)?.to_vec1::<f32>()?;

    // Use helper to compare interleaved complex sinusoid.
    assert_approx_scaled(&host, &expected_sin_cos(n), FFT_EPS_COMPLEX);
    Ok(())
}
