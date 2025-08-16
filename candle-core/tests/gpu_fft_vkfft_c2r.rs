#![cfg(all(
    feature = "cuda",
    feature = "gpu-fft",
    feature = "gpu-fft-vkfft",
    feature = "gpu-fft-vkfft-ffi"
))]

use candle_core::{DType, Device, Result, Tensor};

#[test]
fn vkfft_c2r_f32_roundtrip_small_last_axis() -> Result<()> {
    // Build a small real signal, rfft -> irfft and compare to original.
    let n = 16usize;
    let batch = 2usize;
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;

    let data: Vec<f32> = (0..batch * n)
        .map(|i| (i as f32).cos() * 0.2 + (i as f32) * 1e-3)
        .collect();
    let t_cpu = Tensor::from_vec(data.clone(), (batch, n), &cpu)?;
    let t_gpu = Tensor::from_vec(data.clone(), (batch, n), &cuda)?;

    // GPU: rfft then irfft (no normalization)
    let spec = t_gpu.rfft(1, false)?;
    let back = spec.irfft(1, false)?;
    let back_cpu = back.to_device(&cpu)?;

    let v_in = t_cpu.flatten_all()?.to_vec1::<f32>()?;
    let v_back = back_cpu.flatten_all()?.to_vec1::<f32>()?;

    // Expect exact size match and close values
    assert_eq!(back.dtype(), DType::F32);
    assert_eq!(back.dims(), t_gpu.dims());
    for (x, y) in v_in.iter().zip(v_back.iter()) {
        assert!((x - y).abs() < 1e-3, "x={x} y={y}");
    }
    Ok(())
}
