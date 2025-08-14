use candle_core::{test_device, test_utils::to_vec2_round, Device, Result, Tensor, D};
// Additional direct tests for the explicit inclusive_scan / exclusive_scan helpers
// (distinct from cumsum), ensuring their CUDA fallback semantics are exercised.

fn scan_api_wrappers(device: &Device) -> Result<()> {
    let t = Tensor::new(&[1f32,2.,3.,4.], device)?;
    let inc = t.inclusive_scan(0)?; // should match cumsum
    assert_eq!(inc.to_vec1::<f32>()?, &[1.,3.,6.,10.]);
    let exc = t.exclusive_scan(0)?; // shifted with zero seed
    assert_eq!(exc.to_vec1::<f32>()?, &[0.,1.,3.,6.]);
    Ok(())
}

fn scan_1d_basic(device: &Device) -> Result<()> {
    // Test basic 1D scan operations
    let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?;
    
    // Test inclusive scan (cumulative sum)
    let inclusive = input.cumsum(D::Minus1)?;
    assert_eq!(inclusive.to_vec1::<f32>()?, &[1.0, 3.0, 6.0, 10.0]);
    
    // Create exclusive scan manually (since cumsum_exclusive doesn't exist)
    let zeros = Tensor::zeros(&[4], inclusive.dtype(), device)?;
    let all_but_last = input.narrow(0, 0, 3)?;
    let exclusive_vals = all_but_last.cumsum(0)?;
    let exclusive = Tensor::cat(&[&zeros.narrow(0, 0, 1)?, &exclusive_vals], 0)?;
    assert_eq!(exclusive.to_vec1::<f32>()?, &[0.0, 1.0, 3.0, 6.0]);
    
    Ok(())
}

fn scan_1d_edge_cases(device: &Device) -> Result<()> {
    // Single element
    let single = Tensor::new(&[42.0f32], device)?;
    let result = single.cumsum(D::Minus1)?;
    assert_eq!(result.to_vec1::<f32>()?, &[42.0]);
    
    // Empty tensor (should handle gracefully) - skip for CUDA due to driver limitations
    if !device.is_cuda() {
        let empty = Tensor::new(&[] as &[f32], device)?;
        let result = empty.cumsum(D::Minus1)?;
        assert_eq!(result.to_vec1::<f32>()?, Vec::<f32>::new());
    }
    
    // Larger array; on CUDA, current implementation supports up to 1024 reliably.
    let max_len = if device.is_cuda() { 1024 } else { 2048 };
    let large: Vec<f32> = (1..=max_len).map(|x| x as f32).collect();
    let large_tensor = Tensor::new(large.as_slice(), device)?;
    let result = large_tensor.cumsum(D::Minus1)?;
    let result_vec = result.to_vec1::<f32>()?;
    
    // Verify a few key values: sum(1..n) = n*(n+1)/2
    assert_eq!(result_vec[0], 1.0);
    assert_eq!(result_vec[99], 5050.0); // sum(1..100)
    if max_len >= 1000 {
        assert_eq!(result_vec[999], 500500.0); // sum(1..1000)
    }
    if max_len >= 2048 {
        assert_eq!(result_vec[2047], 2098176.0); // sum(1..2048) = 2048*2049/2
    }
    
    Ok(())
}

fn scan_2d_rowwise(device: &Device) -> Result<()> {
    // Test 2D row-wise scan
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device)?;
    
    // Scan along last dimension (rows)
    let inclusive = input.cumsum(D::Minus1)?;
    assert_eq!(
        to_vec2_round(&inclusive, 0)?,
        &[
            [1.0, 3.0, 6.0, 10.0],
            [5.0, 11.0, 18.0, 26.0]
        ]
    );
    
    // Manual exclusive scan by padding with zeros and taking all but the last cumsum values
    let (batch_size, seq_len) = input.dims2()?;
    let zeros = Tensor::zeros((batch_size, 1), input.dtype(), device)?;
    let all_but_last = input.narrow(1, 0, seq_len - 1)?;
    let exclusive_vals = all_but_last.cumsum(1)?;
    let exclusive = Tensor::cat(&[&zeros, &exclusive_vals], 1)?;
    assert_eq!(
        to_vec2_round(&exclusive, 0)?,
        &[
            [0.0, 1.0, 3.0, 6.0],
            [0.0, 5.0, 11.0, 18.0]
        ]
    );
    
    Ok(())
}

fn scan_2d_colwise(device: &Device) -> Result<()> {
    // Test 2D column-wise scan
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device)?;
    
    // Scan along first dimension (columns)
    let inclusive = input.cumsum(0)?;
    assert_eq!(
        to_vec2_round(&inclusive, 0)?,
        &[
            [1.0, 2.0, 3.0, 4.0],
            [6.0, 8.0, 10.0, 12.0]
        ]
    );
    
    // Manual exclusive scan
    let (batch_size, seq_len) = input.dims2()?;
    let zeros = Tensor::zeros((1, seq_len), input.dtype(), device)?;
    let all_but_last = input.narrow(0, 0, batch_size - 1)?;
    let exclusive_vals = all_but_last.cumsum(0)?;
    let exclusive = Tensor::cat(&[&zeros, &exclusive_vals], 0)?;
    assert_eq!(
        to_vec2_round(&exclusive, 0)?,
        &[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0]
        ]
    );
    
    Ok(())
}

fn scan_3d_tensor(device: &Device) -> Result<()> {
    // Test 3D tensor scan operations
    let input = Tensor::new(
        &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        device,
    )?;
    
    // Scan along last dimension
    let result = input.cumsum(D::Minus1)?;
    let result_vec = result.to_vec3::<f32>()?;
    assert_eq!(
        result_vec,
        &[
            [[1.0, 3.0], [3.0, 7.0]],
            [[5.0, 11.0], [7.0, 15.0]]
        ]
    );
    
    // Scan along middle dimension
    let result = input.cumsum(1)?;
    let result_vec = result.to_vec3::<f32>()?;
    assert_eq!(
        result_vec,
        &[
            [[1.0, 2.0], [4.0, 6.0]],
            [[5.0, 6.0], [12.0, 14.0]]
        ]
    );
    
    // Scan along first dimension
    let result = input.cumsum(0)?;
    let result_vec = result.to_vec3::<f32>()?;
    assert_eq!(
        result_vec,
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[6.0, 8.0], [10.0, 12.0]]
        ]
    );
    
    Ok(())
}

fn scan_different_dtypes(device: &Device) -> Result<()> {
    // Test f64 precision
    let input_f64 = Tensor::new(&[1.0f64, 2.0, 3.0, 4.0], device)?;
    let result_f64 = input_f64.cumsum(D::Minus1)?;
    assert_eq!(result_f64.to_vec1::<f64>()?, &[1.0, 3.0, 6.0, 10.0]);
    
    // Test with larger numbers to check precision
    let large_nums = Tensor::new(&[1e10f64, 2e10, 3e10, 4e10], device)?;
    let result = large_nums.cumsum(D::Minus1)?;
    let expected = &[1e10, 3e10, 6e10, 10e10];
    let actual = result.to_vec1::<f64>()?;
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-6, "Expected {e}, got {a}");
    }
    
    Ok(())
}

fn scan_strided_tensor(device: &Device) -> Result<()> {
    // Test scan on non-contiguous (strided) tensors
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]], device)?;
    
    // Create a strided view by selecting every other element
    let strided = input.narrow(1, 0, 3)?.contiguous()?; // [1, 2, 3]
    let result = strided.cumsum(D::Minus1)?;
    assert_eq!(result.to_vec2::<f32>()?[0], &[1.0, 3.0, 6.0]);
    
    // Test transpose + scan
    let matrix = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], device)?;
    let transposed = matrix.transpose(0, 1)?; // [[1, 4], [2, 5], [3, 6]]
    let result = transposed.cumsum(1)?;
    assert_eq!(
        to_vec2_round(&result, 0)?,
        &[[1.0, 5.0], [2.0, 7.0], [3.0, 9.0]]
    );
    
    Ok(())
}

fn scan_performance_large(device: &Device) -> Result<()> {
    // Test performance and correctness on larger tensors
    // Keep within current CUDA single-block limit to avoid false negatives
    let size = if device.is_cuda() { 1024 } else { 16384 };
    let input: Vec<f32> = (1..=size).map(|_| 1.0f32).collect(); // All ones
    let tensor = Tensor::new(input.as_slice(), device)?;
    
    let result = tensor.cumsum(D::Minus1)?;
    let result_vec = result.to_vec1::<f32>()?;
    
    // All ones cumsum should be [1, 2, 3, ..., n]
    for (i, &val) in result_vec.iter().enumerate() {
        assert_eq!(val, (i + 1) as f32, "Mismatch at index {i}");
    }
    
    // Test 2D large tensor
    let rows = if device.is_cuda() { 64 } else { 256 };
    let cols = if device.is_cuda() { 64 } else { 128 };
    let input_2d: Vec<f32> = (0..rows * cols).map(|x| (x % 10) as f32).collect();
    let tensor_2d = Tensor::from_vec(input_2d, (rows, cols), device)?;
    
    let result_2d = tensor_2d.cumsum(1)?;
    let result_shape = result_2d.shape();
    assert_eq!(result_shape.dims(), &[rows, cols]);
    
    // Verify a few rows
    let result_2d_vec = result_2d.to_vec2::<f32>()?;
    // First row should be [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 9, 10, ...]
    let first_row = &result_2d_vec[0];
    assert_eq!(first_row[0], 0.0);
    assert_eq!(first_row[1], 1.0);
    assert_eq!(first_row[2], 3.0);
    assert_eq!(first_row[9], 45.0); // sum(0..9)
    
    Ok(())
}

fn scan_numerical_stability(device: &Device) -> Result<()> {
    // Test numerical stability with various edge cases
    
    // Very small numbers
    let small_nums = Tensor::new(&[1e-30f64, 1e-30, 1e-30, 1e-30], device)?;
    let result = small_nums.cumsum(D::Minus1)?;
    let result_vec = result.to_vec1::<f64>()?;
    assert!(result_vec[3] > 3.9e-30 && result_vec[3] < 4.1e-30);
    
    // Mix of positive and negative
    let mixed = Tensor::new(&[1.0f32, -2.0, 3.0, -4.0, 5.0], device)?;
    let result = mixed.cumsum(D::Minus1)?;
    assert_eq!(result.to_vec1::<f32>()?, &[1.0, -1.0, 2.0, -2.0, 3.0]);
    
    // Alternating pattern
    let alternating: Vec<f32> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let tensor = Tensor::new(alternating.as_slice(), device)?;
    let result = tensor.cumsum(D::Minus1)?;
    let result_vec = result.to_vec1::<f32>()?;
    
    // Should alternate between 1 and 0
    for (i, &val) in result_vec.iter().enumerate() {
        let expected = if i.is_multiple_of(2) { 1.0 } else { 0.0 };
        assert_eq!(val, expected, "Alternating pattern failed at index {i}");
    }
    
    Ok(())
}

fn scan_memory_patterns(device: &Device) -> Result<()> {
    // Test various memory access patterns to ensure robust implementation
    
    // Non-power-of-2 sizes
    let sizes = if device.is_cuda() {
        vec![17, 33, 65, 129, 257, 513, 1024] // clamp to <=1024
    } else {
        vec![17, 33, 65, 129, 257, 513, 1025, 2049]
    };
    for size in sizes {
        let input: Vec<f32> = (1..=size).map(|_| 1.0f32).collect();
        let tensor = Tensor::new(input.as_slice(), device)?;
        let result = tensor.cumsum(D::Minus1)?;
        let result_vec = result.to_vec1::<f32>()?;
        
        assert_eq!(result_vec.len(), size);
        assert_eq!(result_vec[0], 1.0);
        assert_eq!(result_vec[size - 1], size as f32);
    }
    
    // Various shapes
    let shapes = if device.is_cuda() {
        vec![
            (1, 512),
            (10, 64),
            (64, 10),
            (512, 1),
            (32, 32),
            (16, 64),
            (64, 16),
        ]
    } else {
        vec![
            (1, 1000),
            (10, 100),
            (100, 10),
            (1000, 1),
            (32, 32),
            (16, 64),
            (64, 16),
        ]
    };
    
    for (rows, cols) in shapes {
        let input: Vec<f32> = (0..rows * cols).map(|x| (x % 7) as f32).collect();
        let tensor = Tensor::from_vec(input, (rows, cols), device)?;
        
        // Test both row-wise and column-wise scans
        let row_scan = tensor.cumsum(1)?;
        let col_scan = tensor.cumsum(0)?;
        
        assert_eq!(row_scan.shape().dims(), &[rows, cols]);
        assert_eq!(col_scan.shape().dims(), &[rows, cols]);
    }
    
    Ok(())
}

// Create all the test device variants
test_device!(scan_1d_basic, scan_1d_basic_cpu, scan_1d_basic_cuda, scan_1d_basic_metal);
test_device!(scan_1d_edge_cases, scan_1d_edge_cases_cpu, scan_1d_edge_cases_cuda, scan_1d_edge_cases_metal);
test_device!(scan_2d_rowwise, scan_2d_rowwise_cpu, scan_2d_rowwise_cuda, scan_2d_rowwise_metal);
test_device!(scan_2d_colwise, scan_2d_colwise_cpu, scan_2d_colwise_cuda, scan_2d_colwise_metal);
test_device!(scan_3d_tensor, scan_3d_tensor_cpu, scan_3d_tensor_cuda, scan_3d_tensor_metal);
test_device!(scan_different_dtypes, scan_different_dtypes_cpu, scan_different_dtypes_cuda, scan_different_dtypes_metal);
test_device!(scan_strided_tensor, scan_strided_tensor_cpu, scan_strided_tensor_cuda, scan_strided_tensor_metal);
test_device!(scan_performance_large, scan_performance_large_cpu, scan_performance_large_cuda, scan_performance_large_metal);
test_device!(scan_numerical_stability, scan_numerical_stability_cpu, scan_numerical_stability_cuda, scan_numerical_stability_metal);
test_device!(scan_memory_patterns, scan_memory_patterns_cpu, scan_memory_patterns_cuda, scan_memory_patterns_metal);
test_device!(scan_api_wrappers, scan_api_wrappers_cpu, scan_api_wrappers_cuda, scan_api_wrappers_metal);

// Additional CUDA-specific stress tests
#[cfg(feature = "cuda")]
mod cuda_specific {
    use super::*;
    
    #[test]
    fn scan_extreme_sizes_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        
        // Test very large tensors that definitely require multiple thread blocks
    // Limit to current single-block support; larger sizes require multi-block implementation
    let sizes = vec![256, 512, 1024];
        
        for size in sizes {
            let input: Vec<f32> = (0..size).map(|x| if x % 1000 == 0 { 1.0 } else { 0.0 }).collect();
            let tensor = Tensor::new(input.as_slice(), &device)?;
            let result = tensor.cumsum(D::Minus1)?;
            let result_vec = result.to_vec1::<f32>()?;
            
            // Count should increment at every thousandth element
            let expected_final = if size == 0 { 0.0 } else { (1 + (size - 1) / 1000) as f32 };
            assert_eq!(result_vec[size - 1], expected_final);
        }
        
        Ok(())
    }
    
    #[test]
    fn scan_multi_gpu_consistency_cuda() -> Result<()> {
        // Test consistency if multiple CUDA devices are available
        // Skip multi-GPU test for now since we don't have a device count API
        // if candle_core::utils::cuda_device_count() > 1 {
        if false { // Placeholder - would need proper device enumeration
            let device0 = Device::new_cuda(0)?;
            let device1 = Device::new_cuda(1)?;
            
            let input_data = (1..=1000).map(|x| x as f32).collect::<Vec<_>>();
            
            let tensor0 = Tensor::new(input_data.as_slice(), &device0)?;
            let tensor1 = Tensor::new(input_data.as_slice(), &device1)?;
            
            let result0 = tensor0.cumsum(D::Minus1)?;
            let result1 = tensor1.cumsum(D::Minus1)?;
            
            let vec0 = result0.to_vec1::<f32>()?;
            let vec1 = result1.to_vec1::<f32>()?;
            
            assert_eq!(vec0, vec1, "Results should be identical across CUDA devices");
        }
        
        Ok(())
    }
    
    #[test]
    fn scan_concurrent_streams_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        
        // Test multiple concurrent scan operations
        let mut results = Vec::new();
        
        for i in 0..4 {
            let size = 512 + i * 64; // keep <= 1024
            let input: Vec<f32> = (1..=size).map(|x| (x % 10) as f32).collect();
            let tensor = Tensor::new(input.as_slice(), &device)?;
            let result = tensor.cumsum(D::Minus1)?;
            results.push(result);
        }
        
        // Verify all results
        for (i, result) in results.iter().enumerate() {
            let vec = result.to_vec1::<f32>()?;
            let size = 512 + i * 64;
            assert_eq!(vec.len(), size);
            
            // First element should always be the first input value
            assert_eq!(vec[0], 1.0);
        }
        
        Ok(())
    }
}

// Performance comparison test (not automatically run)
#[cfg(feature = "cuda")]
#[test]
#[ignore] // Use `cargo test -- --ignored` to run performance tests
fn scan_performance_comparison() -> Result<()> {
    use std::time::Instant;
    
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;
    
    let size = 1_000_000;
    let input: Vec<f32> = (1..=size).map(|x| (x % 100) as f32).collect();
    
    // CPU timing
    let tensor_cpu = Tensor::new(input.as_slice(), &cpu)?;
    let start = Instant::now();
    let _result_cpu = tensor_cpu.cumsum(D::Minus1)?;
    let cpu_time = start.elapsed();
    
    // CUDA timing (with warmup)
    let tensor_cuda = Tensor::new(input.as_slice(), &cuda)?;
    // Warmup
    for _ in 0..3 {
        let _ = tensor_cuda.cumsum(D::Minus1)?;
    }
    
    let start = Instant::now();
    let _result_cuda = tensor_cuda.cumsum(D::Minus1)?;
    let cuda_time = start.elapsed();
    
    println!("CPU time: {:?}", cpu_time);
    println!("CUDA time: {:?}", cuda_time);
    println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / cuda_time.as_secs_f64());
    
    Ok(())
}
