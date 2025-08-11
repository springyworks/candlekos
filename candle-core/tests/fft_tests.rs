use candle_core::{Device, Result, Tensor, DType};
use std::f32::consts::PI;

#[test]
fn test_cpu_fft_basic() -> Result<()> {
    let device = Device::Cpu;
    
    // Test with a simple sine wave
    let n = 64;
    let freq = 5.0;
    let data: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / n as f32).sin())
        .collect();
    
    let tensor = Tensor::from_vec(data, &[n], &device)?;
    
    // Compute FFT
    let fft_result = tensor.rfft(0, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    
    // Check that we have the expected shape
    assert_eq!(fft_result.dims(), &[n + 2]); // Real-to-complex FFT output
    assert_eq!(magnitude.dims(), &[n / 2 + 1]);
    
    // The peak should be at frequency bin 5 (freq = 5)
    let mag_data = magnitude.to_vec1::<f32>()?;
    let peak_idx = mag_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    
    assert_eq!(peak_idx, freq as usize, "Peak should be at frequency bin {}", freq);
    
    Ok(())
}

#[test]
fn test_cpu_fft_2d() -> Result<()> {
    let device = Device::Cpu;
    
    // Create a 2D signal with known frequency content
    let h = 32;
    let w = 32;
    let data: Vec<f32> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let fx = 3.0;
                let fy = 2.0;
                (2.0 * PI * fx * x as f32 / w as f32).sin() * 
                (2.0 * PI * fy * y as f32 / h as f32).cos()
            })
        })
        .collect();
    
    let tensor = Tensor::from_vec(data, &[h, w], &device)?;
    
    // Compute 2D FFT
    let fft_result = tensor.fft2(true, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    
    // Check dimensions
    assert_eq!(fft_result.dims(), &[h * 2, w + 2]); // 2D real-to-complex FFT
    assert_eq!(magnitude.dims(), &[h, w / 2 + 1]);
    
    Ok(())
}

#[test]
fn test_cpu_fft_inverse() -> Result<()> {
    let device = Device::Cpu;
    
    // Create test data
    let n = 128;
    let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let original = Tensor::from_vec(data, &[n], &device)?;
    
    // Forward FFT then inverse FFT
    let fft_result = original.fft(0, false, true)?; // Complex input, normalized
    let reconstructed = fft_result.ifft(0, true)?;
    
    // Extract real part (assuming we started with real data)
    let real_part = reconstructed.narrow(0, 0, n)?;
    
    // Check that we get back approximately the original
    let orig_data = original.to_vec1::<f32>()?;
    let recon_data = real_part.to_vec1::<f32>()?;
    
    for (i, (&orig, &recon)) in orig_data.iter().zip(recon_data.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-5,
            "Mismatch at index {}: original={}, reconstructed={}",
            i, orig, recon
        );
    }
    
    Ok(())
}

#[test]
fn test_cpu_fft_windowing() -> Result<()> {
    let device = Device::Cpu;
    
    // Test that windowing reduces spectral leakage
    let n = 256;
    let freq = 10.5; // Non-integer frequency to cause leakage
    
    let data: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / n as f32).sin())
        .collect();
    
    let tensor = Tensor::from_vec(data, &[n], &device)?;
    
    // Apply Hann window (this would need to be implemented in the FFT backend)
    // For now, just test that FFT works with the data
    let fft_result = tensor.rfft(0, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    
    assert_eq!(magnitude.dims(), &[n / 2 + 1]);
    
    // The magnitude should have reasonable values
    let mag_data = magnitude.to_vec1::<f32>()?;
    assert!(mag_data.iter().all(|&x| x >= 0.0 && x.is_finite()));
    
    Ok(())
}

#[test]
fn test_cpu_fft_different_sizes() -> Result<()> {
    let device = Device::Cpu;
    
    // Test various FFT sizes
    let sizes = vec![16, 32, 64, 128, 256, 512];
    
    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(data, &[size], &device)?;
        
        let fft_result = tensor.rfft(0, false)?;
        let magnitude = fft_result.fft_magnitude()?;
        
        assert_eq!(magnitude.dims(), &[size / 2 + 1]);
        
        // Check that magnitudes are reasonable
        let mag_data = magnitude.to_vec1::<f32>()?;
        assert!(mag_data.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }
    
    Ok(())
}

#[test]
fn test_cpu_fft_multidimensional() -> Result<()> {
    let device = Device::Cpu;
    
    // Test FFT on higher-dimensional tensors
    let batch_size = 4;
    let seq_len = 64;
    
    let data: Vec<f32> = (0..batch_size * seq_len)
        .map(|i| ((i % seq_len) as f32 * 0.1).sin())
        .collect();
    
    let tensor = Tensor::from_vec(data, &[batch_size, seq_len], &device)?;
    
    // FFT along the last dimension
    let fft_result = tensor.rfft(1, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    
    assert_eq!(magnitude.dims(), &[batch_size, seq_len / 2 + 1]);
    
    Ok(())
}

#[test]
fn test_cpu_fft_dtype_support() -> Result<()> {
    let device = Device::Cpu;
    
    // Test f32
    let data_f32: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
    let tensor_f32 = Tensor::from_vec(data_f32, &[64], &device)?;
    let fft_f32 = tensor_f32.rfft(0, false)?;
    assert_eq!(fft_f32.dtype(), DType::F32);
    
    // Test f64 (should work via conversion)
    let data_f64: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
    let tensor_f64 = Tensor::from_vec(data_f64, &[64], &device)?;
    let fft_f64 = tensor_f64.rfft(0, false)?;
    assert_eq!(fft_f64.dtype(), DType::F64);
    
    Ok(())
}

#[test]
fn test_cpu_fft_magnitude_phase() -> Result<()> {
    let device = Device::Cpu;
    
    // Create complex signal (sin + cos with phase shift)
    let n = 128;
    let freq = 8.0;
    let phase_shift = PI / 4.0;
    
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let t = 2.0 * PI * freq * i as f32 / n as f32;
            (t + phase_shift).sin()
        })
        .collect();
    
    let tensor = Tensor::from_vec(data, &[n], &device)?;
    let fft_result = tensor.rfft(0, false)?;
    
    // Test magnitude and phase extraction
    let magnitude = fft_result.fft_magnitude()?;
    let phase = fft_result.fft_phase()?;
    
    assert_eq!(magnitude.dims(), &[n / 2 + 1]);
    assert_eq!(phase.dims(), &[n / 2 + 1]);
    
    // Check that magnitude and phase values are reasonable
    let mag_data = magnitude.to_vec1::<f32>()?;
    let phase_data = phase.to_vec1::<f32>()?;
    
    assert!(mag_data.iter().all(|&x| x >= 0.0 && x.is_finite()));
    assert!(phase_data.iter().all(|&x| x >= -PI && x <= PI && x.is_finite()));
    
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_fft_basic() -> Result<()> {
    let device = Device::new_cuda(0)?;
    
    // Test with a simple sine wave
    let n = 64;
    let freq = 5.0;
    let data: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / n as f32).sin())
        .collect();
    
    let tensor = Tensor::from_vec(data, &[n], &device)?;
    
    // Compute FFT
    let fft_result = tensor.rfft(0, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    
    // Check that we have the expected shape
    assert_eq!(fft_result.dims(), &[n + 2]);
    assert_eq!(magnitude.dims(), &[n / 2 + 1]);
    
    // The peak should be at frequency bin 5
    let mag_data = magnitude.to_vec1::<f32>()?;
    let peak_idx = mag_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    
    assert_eq!(peak_idx, freq as usize);
    
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_cpu_fft_consistency() -> Result<()> {
    let cpu_device = Device::Cpu;
    let cuda_device = Device::new_cuda(0)?;
    
    // Create test data
    let n = 128;
    let data: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 3.0 * i as f32 / n as f32).sin())
        .collect();
    
    // Compute FFT on both devices
    let cpu_tensor = Tensor::from_vec(data.clone(), &[n], &cpu_device)?;
    let cuda_tensor = Tensor::from_vec(data, &[n], &cuda_device)?;
    
    let cpu_fft = cpu_tensor.rfft(0, false)?;
    let cuda_fft = cuda_tensor.rfft(0, false)?;
    
    // Move CUDA result to CPU for comparison
    let cuda_fft_cpu = cuda_fft.to_device(&cpu_device)?;
    
    // Compare results
    let cpu_data = cpu_fft.to_vec1::<f32>()?;
    let cuda_data = cuda_fft_cpu.to_vec1::<f32>()?;
    
    for (i, (&cpu_val, &cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
        assert!(
            (cpu_val - cuda_val).abs() < 1e-4,
            "Mismatch at index {}: CPU={}, CUDA={}",
            i, cpu_val, cuda_val
        );
    }
    
    Ok(())
}

#[test]
fn test_fft_error_handling() -> Result<()> {
    let device = Device::Cpu;
    
    // Test with unsupported dtype
    let data = vec![1i32, 2, 3, 4];
    let tensor = Tensor::from_vec(data, &[4], &device)?;
    
    let result = tensor.rfft(0, false);
    assert!(result.is_err());
    
    // Test with invalid dimension
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data, &[4], &device)?;
    
    let result = tensor.rfft(2, false); // Invalid dimension
    assert!(result.is_err());
    
    Ok(())
}

#[test]
fn test_fft_performance_vs_complexity() -> Result<()> {
    let device = Device::Cpu;
    
    // Test that larger FFTs still complete in reasonable time
    let sizes = vec![512, 1024, 2048];
    
    for &size in &sizes {
        let start = std::time::Instant::now();
        
        let data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(data, &[size], &device)?;
        let _fft_result = tensor.rfft(0, false)?;
        
        let duration = start.elapsed();
        println!("FFT of size {} took: {:?}", size, duration);
        
        // Ensure it completes within a reasonable time (10 seconds)
        assert!(duration.as_secs() < 10, "FFT took too long: {:?}", duration);
    }
    
    Ok(())
}
