#![cfg(all(
    feature = "cuda",
    feature = "gpu-fft",
    feature = "gpu-fft-vkfft",
    feature = "gpu-fft-vkfft-ffi"
))]

use candle_core::{DType, Device, Result, Tensor};

fn make_complex_interleaved(real: &[f32], imag: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(real.len() * 2);
    for i in 0..real.len() {
        out.push(real[i]);
        out.push(imag[i]);
    }
    out
}

#[test]
fn vkfft_c2c_f32_roundtrip_last_axis() -> Result<()> {
    // Create simple complex data and verify c2c forward+inverse returns original
    let n_complex = 8usize; // complex length
    let batch = 2usize;
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;

    // Build complex signal as interleaved real/imag floats
    let mut real = Vec::with_capacity(batch * n_complex);
    let mut imag = Vec::with_capacity(batch * n_complex);
    for i in 0..(batch * n_complex) {
        real.push((i as f32).sin() * 0.1);
        imag.push((i as f32).cos() * 0.05);
    }
    let inter = make_complex_interleaved(&real, &imag);

    // Shape uses float count along last axis (2 * n_complex)
    let t = Tensor::from_vec(inter.clone(), (batch, n_complex * 2), &cuda)?;

    // Forward complex (treat input as complex) then inverse
    let spec = t.fft(1, false, false)?; // forward complex-to-complex (no normalization)
    let back = spec.ifft(1, true)?; // inverse complex-to-complex (normalize to recover original)

    let back_cpu = back.to_device(&cpu)?;
    let v_back = back_cpu.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(back.dtype(), DType::F32);
    assert_eq!(back.dims(), t.dims());

    for (x, y) in inter.iter().zip(v_back.iter()) {
        assert!((x - y).abs() < 1e-3, "x={x} y={y}");
    }

    Ok(())
}
