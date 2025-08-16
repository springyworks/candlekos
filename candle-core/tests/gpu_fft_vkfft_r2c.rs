#![cfg(all(
    feature = "cuda",
    feature = "gpu-fft",
    feature = "gpu-fft-vkfft",
    feature = "gpu-fft-vkfft-ffi"
))]

use candle_core::{DType, Device, Result, Tensor};

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > tol {
            return false;
        }
    }
    true
}

#[test]
fn vkfft_r2c_f32_matches_cpu_small_last_axis() -> Result<()> {
    // Use last axis for simplicity to match current provider impl.
    let n = 16usize;
    let batch = 3usize;
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;

    // Deterministic input
    let data: Vec<f32> = (0..batch * n)
        .map(|i| (i as f32).sin() * 0.1 + (i as f32) * 1e-3)
        .collect();
    let t_cpu = Tensor::from_vec(data.clone(), (batch, n), &cpu)?;
    let t_gpu = Tensor::from_vec(data, (batch, n), &cuda)?;

    // Real-to-complex fft on last dim
    let cpu_fft = t_cpu.rfft(1, false)?; // not normalized
    let gpu_fft = t_gpu.rfft(1, false)?;

    // Move gpu result back to cpu and compare
    let gpu_fft_cpu = gpu_fft.to_device(&cpu)?;
    let v_cpu = cpu_fft.flatten_all()?.to_vec1::<f32>()?;
    let v_gpu = gpu_fft_cpu.flatten_all()?.to_vec1::<f32>()?;

    // Loose-ish tolerance (VkFFT vs rustfft minor diffs)
    assert!(
        approx_eq(&v_cpu, &v_gpu, 1e-3),
        "mismatch\nCPU: {:?}\nGPU: {:?}",
        &v_cpu[..8.min(v_cpu.len())],
        &v_gpu[..8.min(v_gpu.len())]
    );
    assert_eq!(cpu_fft.dtype(), DType::F32);
    assert_eq!(gpu_fft.dtype(), DType::F32);
    assert_eq!(cpu_fft.dims(), gpu_fft.dims());
    Ok(())
}
