use candle_core::{Device, Result, Tensor, DType};
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("Candle FFT Demo");
    println!("===============");
    
    // Create a device (try CUDA first, fall back to CPU)
    let device = if candle_core::utils::cuda_is_available() {
        println!("Using CUDA device");
        Device::new_cuda(0)?
    } else {
        println!("Using CPU device");
        Device::Cpu
    };
    
    // Demo 1: Basic 1D FFT on a sine wave
    demo_basic_fft(&device)?;
    
    // Demo 2: 2D FFT for image-like data
    demo_2d_fft(&device)?;
    
    // Demo 3: Frequency analysis
    demo_frequency_analysis(&device)?;
    
    // Demo 4: FFT-based filtering
    demo_fft_filtering(&device)?;
    
    // Demo 5: Performance comparison
    demo_performance(&device)?;
    
    Ok(())
}

fn demo_basic_fft(device: &Device) -> Result<()> {
    println!("\n1. Basic 1D FFT Demo");
    println!("--------------------");
    
    // Create a composite signal: 5 Hz + 15 Hz + 25 Hz
    let n = 256;
    let sample_rate = 100.0; // Hz
    
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * PI * 5.0 * t).sin() +
            0.5 * (2.0 * PI * 15.0 * t).sin() +
            0.25 * (2.0 * PI * 25.0 * t).sin()
        })
        .collect();
    
    let tensor = Tensor::from_vec(signal, &[n], device)?;
    
    // Compute FFT
    let fft_result = tensor.rfft(0, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    let phase = fft_result.fft_phase()?;
    
    println!("Signal length: {}", n);
    println!("FFT output shape: {:?}", fft_result.dims());
    println!("Magnitude shape: {:?}", magnitude.dims());
    println!("Phase shape: {:?}", phase.dims());
    
    // Find the top 3 frequency components
    let mag_data = magnitude.to_vec1::<f32>()?;
    let mut freq_mag: Vec<(usize, f32)> = mag_data
        .iter()
        .enumerate()
        .map(|(i, &mag)| (i, mag))
        .collect();
    
    freq_mag.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Top 3 frequency components:");
    for (i, (freq_bin, magnitude)) in freq_mag.iter().take(3).enumerate() {
        let frequency = *freq_bin as f32 * sample_rate / n as f32;
        println!("  {}. Bin {}: {:.2} Hz, Magnitude: {:.3}", 
                 i + 1, freq_bin, frequency, magnitude);
    }
    
    Ok(())
}

fn demo_2d_fft(device: &Device) -> Result<()> {
    println!("\n2. 2D FFT Demo (Image-like Data)");
    println!("--------------------------------");
    
    // Create a 2D pattern with known spatial frequencies
    let height = 64;
    let width = 64;
    
    let image: Vec<f32> = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                // Create a pattern with spatial frequencies
                let fx = 3.0; // cycles per image width
                let fy = 2.0; // cycles per image height
                
                let pattern1 = (2.0 * PI * fx * x as f32 / width as f32).sin();
                let pattern2 = (2.0 * PI * fy * y as f32 / height as f32).cos();
                
                pattern1 * pattern2 + 0.1 * ((x + y) as f32 / 10.0).sin()
            })
        })
        .collect();
    
    let tensor = Tensor::from_vec(image, &[height, width], device)?;
    
    // Compute 2D FFT
    let fft_result = tensor.fft2(true, false)?;
    let magnitude = fft_result.fft_magnitude()?;
    
    println!("Image size: {}x{}", height, width);
    println!("2D FFT output shape: {:?}", fft_result.dims());
    println!("2D Magnitude shape: {:?}", magnitude.dims());
    
    // Find the peak in the frequency domain
    let mag_data = magnitude.to_vec1::<f32>()?;
    let max_magnitude = mag_data.iter().fold(0.0f32, |acc, &x| acc.max(x));
    println!("Maximum magnitude in 2D FFT: {:.3}", max_magnitude);
    
    Ok(())
}

fn demo_frequency_analysis(device: &Device) -> Result<()> {
    println!("\n3. Frequency Analysis Demo");
    println!("-------------------------");
    
    // Simulate a signal with noise
    let n = 512;
    let sample_rate = 1000.0;
    
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            // Clean signal: 50 Hz and 150 Hz
            let clean = (2.0 * PI * 50.0 * t).sin() + 0.5 * (2.0 * PI * 150.0 * t).sin();
            // Add some noise
            let noise = 0.1 * (t * 1000.0).sin();
            clean + noise
        })
        .collect();
    
    let tensor = Tensor::from_vec(signal, &[n], device)?;
    
    // Compute FFT
    let fft_result = tensor.rfft(0, true)?; // Normalized
    let magnitude = fft_result.fft_magnitude()?;
    
    // Calculate frequency bins
    let mag_data = magnitude.to_vec1::<f32>()?;
    let num_bins = mag_data.len();
    
    println!("Frequency analysis of noisy signal:");
    println!("Sample rate: {} Hz", sample_rate);
    println!("Number of frequency bins: {}", num_bins);
    
    // Find significant frequency components (above threshold)
    let threshold = 0.1 * mag_data.iter().fold(0.0f32, |acc, &x| acc.max(x));
    
    println!("Significant frequency components (magnitude > {:.3}):", threshold);
    for (bin, &magnitude) in mag_data.iter().enumerate() {
        if magnitude > threshold {
            let frequency = bin as f32 * sample_rate / n as f32;
            println!("  {:.1} Hz: magnitude {:.3}", frequency, magnitude);
        }
    }
    
    Ok(())
}

fn demo_fft_filtering(device: &Device) -> Result<()> {
    println!("\n4. FFT-based Filtering Demo");
    println!("---------------------------");
    
    let n = 256;
    let sample_rate = 100.0;
    
    // Create a signal with low and high frequency components
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            // Low frequency component (2 Hz)
            let low_freq = (2.0 * PI * 2.0 * t).sin();
            // High frequency component (30 Hz)
            let high_freq = 0.5 * (2.0 * PI * 30.0 * t).sin();
            low_freq + high_freq
        })
        .collect();
    
    let tensor = Tensor::from_vec(signal, &[n], device)?;
    
    // Compute FFT
    let fft_result = tensor.rfft(0, false)?;
    let magnitude_before = fft_result.fft_magnitude()?;
    
    // Simulate low-pass filtering by zeroing high frequencies
    // (In practice, you'd modify the FFT coefficients)
    let mag_before = magnitude_before.to_vec1::<f32>()?;
    let cutoff_bin = (10.0 * n as f32 / sample_rate) as usize; // 10 Hz cutoff
    
    println!("Original signal has {} samples", n);
    println!("Applying low-pass filter with cutoff at bin {} (≈10 Hz)", cutoff_bin);
    
    // Show frequency content before filtering
    println!("\nFrequency content before filtering:");
    for (bin, &mag) in mag_before.iter().enumerate() {
        if mag > 0.1 {
            let freq = bin as f32 * sample_rate / n as f32;
            println!("  {:.1} Hz: {:.3}", freq, mag);
        }
    }
    
    // Compute inverse FFT (this would be the filtered signal)
    let reconstructed = fft_result.ifft(0, false)?;
    println!("\nReconstructed signal shape: {:?}", reconstructed.dims());
    
    Ok(())
}

fn demo_performance(device: &Device) -> Result<()> {
    println!("\n5. Performance Demo");
    println!("------------------");
    
    let sizes = vec![128, 256, 512, 1024, 2048];
    
    println!("Benchmarking FFT performance on {}:", 
             if device.is_cuda() { "CUDA" } else { "CPU" });
    
    for &size in &sizes {
        // Create test data
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let tensor = Tensor::from_vec(data, &[size], device)?;
        
        // Time the FFT computation
        let start = std::time::Instant::now();
        
        // Run multiple iterations for better timing
        let iterations = if size >= 2048 { 10 } else { 100 };
        
        for _ in 0..iterations {
            let _fft_result = tensor.rfft(0, false)?;
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_micros() as f64 / iterations as f64;
        
        println!("  Size {}: {:.2} μs per FFT", size, avg_time);
    }
    
    // Memory usage demo
    let large_size = 4096;
    let data: Vec<f32> = (0..large_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let tensor = Tensor::from_vec(data, &[large_size], device)?;
    
    println!("\nMemory usage example (size {}):", large_size);
    println!("  Input tensor size: {} elements", tensor.elem_count());
    
    let fft_result = tensor.rfft(0, false)?;
    println!("  FFT output size: {} elements", fft_result.elem_count());
    
    let magnitude = fft_result.fft_magnitude()?;
    println!("  Magnitude size: {} elements", magnitude.elem_count());
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn demo_cuda_vs_cpu() -> Result<()> {
    println!("\n6. CUDA vs CPU Comparison");
    println!("-------------------------");
    
    let size = 2048;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    
    // CPU timing
    let cpu_device = Device::Cpu;
    let cpu_tensor = Tensor::from_vec(data.clone(), &[size], &cpu_device)?;
    
    let start = std::time::Instant::now();
    let _cpu_result = cpu_tensor.rfft(0, false)?;
    let cpu_time = start.elapsed();
    
    // CUDA timing
    let cuda_device = Device::new_cuda(0)?;
    let cuda_tensor = Tensor::from_vec(data, &[size], &cuda_device)?;
    
    let start = std::time::Instant::now();
    let _cuda_result = cuda_tensor.rfft(0, false)?;
    let cuda_time = start.elapsed();
    
    println!("FFT performance comparison (size {}):", size);
    println!("  CPU:  {:?}", cpu_time);
    println!("  CUDA: {:?}", cuda_time);
    
    if cuda_time < cpu_time {
        let speedup = cpu_time.as_nanos() as f64 / cuda_time.as_nanos() as f64;
        println!("  CUDA is {:.2}x faster", speedup);
    } else {
        let slowdown = cuda_time.as_nanos() as f64 / cpu_time.as_nanos() as f64;
        println!("  CPU is {:.2}x faster", slowdown);
    }
    
    Ok(())
}
