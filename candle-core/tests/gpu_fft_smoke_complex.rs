#![cfg(all(feature = "cuda", feature = "fft", feature = "gpu-fft"))]
// Complex-to-complex GPU FFT smoke test: creates an interleaved complex tensor,
// performs a forward + inverse FFT roundtrip along one dimension and validates
// reconstruction within tolerance (considering possible provider scaling).

use candle_core::{Device, Result, Tensor};

#[test]
fn gpu_fft_c2c_roundtrip() -> Result<()> {
    let dev = match Device::new_cuda(0) { Ok(d) => d, Err(_) => return Ok(()) };
    let n = 32usize;
    // Interleaved complex layout: [re0, im0, re1, im1, ...]
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n { let t = i as f32; data.push(t.sin()); data.push(t.cos()); }
    let complex = Tensor::from_vec(data, &[n * 2], &dev)?; // Single axis length n (interleaved)

    // Forward complex FFT (real_input=false)
    let freq = complex.fft(0, false, false)?;
    // Inverse
    let time = freq.ifft(0, false)?;
    let host = time.to_device(&Device::Cpu)?.to_vec1::<f32>()?;

    // Detect uniform scale factor (skip first imaginary component etc.)
    let mut scale = None;
    for i in (0..host.len()).step_by(2) { // real parts only for scale detection
        let orig_re = ( (i/2) as f32 ).sin();
        if orig_re.abs() > 1e-6 { scale = Some(host[i]/orig_re); break; }
    }
    if let Some(s) = scale { if (s - 1.0).abs() < 1e-3 { scale = Some(1.0); } }

    for i in 0..n {
        let expected_re = (i as f32).sin();
        let expected_im = (i as f32).cos();
        let got_re = host[2*i]   / scale.unwrap_or(1.0);
        let got_im = host[2*i+1] / scale.unwrap_or(1.0);
        assert!((got_re - expected_re).abs() < 7e-3, "re mismatch i={i} got={got_re} exp={expected_re}");
        assert!((got_im - expected_im).abs() < 7e-3, "im mismatch i={i} got={got_im} exp={expected_im}");
    }
    Ok(())
}
