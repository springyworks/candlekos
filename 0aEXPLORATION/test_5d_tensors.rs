//! 5D Tensor Capabilities Test - Comprehensive testing of Candle's native 5D tensor support
//! Tests creation, operations, transformations, indexing, broadcasting, and performance of 5D tensors on GPU

//! Comprehensive 5D tensor capabilities demonstration in Candle
//! Tests creation, operations, indexing, broadcasting, and performance with video-like tensor dimensions

// 5D Tensor examples and capabilities in Candle
use candle_core::{Device, Result, Tensor, DType, IndexOp};

fn main() -> Result<()> {
    println!("🚀 Testing 5D Tensor capabilities in Candle");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Test 1: Create 5D tensors directly
    println!("\n📦 Test 1: Creating 5D tensors");
    
    // Create a 5D tensor with shape [batch, channels, depth, height, width]
    let shape_5d = vec![2, 3, 4, 8, 8]; // Batch=2, Channels=3, Depth=4, Height=8, Width=8
    let data: Vec<f32> = (0..shape_5d.iter().product::<usize>()).map(|i| i as f32).collect();
    
    let tensor_5d = Tensor::from_vec(data, shape_5d.clone(), &device)?;
    println!("5D Tensor created: {:?}", tensor_5d.shape());
    println!("Rank: {}", tensor_5d.rank());
    println!("Element count: {}", tensor_5d.elem_count());
    
    // Test 2: 5D convolution-like operations
    println!("\n🧮 Test 2: 5D operations");
    
    // Create another 5D tensor for operations
    let tensor_5d_b = Tensor::ones(shape_5d.clone(), DType::F32, &device)?;
    
    // Element-wise operations
    let sum_5d = (&tensor_5d + &tensor_5d_b)?;
    let product_5d = (&tensor_5d * 2.0)?;
    
    println!("Sum shape: {:?}", sum_5d.shape());
    println!("Product shape: {:?}", product_5d.shape());
    
    // Test 3: Reshaping and permuting 5D tensors
    println!("\n🔄 Test 3: Reshaping and permuting");
    
    // Permute dimensions [B, C, D, H, W] -> [B, D, C, H, W]
    let permuted = tensor_5d.permute((0, 2, 1, 3, 4))?;
    println!("Original: {:?}", tensor_5d.shape());
    println!("Permuted: {:?}", permuted.shape());
    
    // Reshape to different 5D configuration
    let reshaped = tensor_5d.reshape((1, 6, 2, 8, 8))?; // Combine batch and channel dims
    println!("Reshaped: {:?}", reshaped.shape());
    
    // Test 4: Slicing 5D tensors
    println!("\n✂️  Test 4: Slicing 5D tensors");
    
    // Get first batch
    let first_batch = tensor_5d.i(0)?; // [3, 4, 8, 8]
    println!("First batch shape: {:?}", first_batch.shape());
    
    // Get specific channel from first batch
    let first_channel = tensor_5d.i((0, 0))?; // [4, 8, 8]
    println!("First channel shape: {:?}", first_channel.shape());
    
    // Multi-dimensional slice
    let slice_5d = tensor_5d.i((..1, ..2, ..2, ..4, ..4))?; // [1, 2, 2, 4, 4]
    println!("5D slice shape: {:?}", slice_5d.shape());
    
    // Test 5: 5D broadcasting
    println!("\n📡 Test 5: Broadcasting with 5D");
    
    // Create tensors for broadcasting
    let tensor_1d = Tensor::ones(8, DType::F32, &device)?; // [8]
    let tensor_3d = Tensor::ones((3, 4, 8), DType::F32, &device)?; // [3, 4, 8]
    
    // Broadcasting examples (reshape to align with 5D tensor)
    let broadcast_1d = tensor_1d.reshape((1, 1, 1, 1, 8))?; // [1, 1, 1, 1, 8]
    let broadcast_3d = tensor_3d.reshape((1, 3, 4, 1, 8))?; // [1, 3, 4, 1, 8]
    
    // Broadcast operations
    let result_1d = (&tensor_5d + &broadcast_1d)?; // Should work
    let result_3d = (&tensor_5d + &broadcast_3d)?; // Should work
    
    println!("5D + 1D broadcast: {:?}", result_1d.shape());
    println!("5D + 3D broadcast: {:?}", result_3d.shape());
    
    // Test 6: 5D convolution simulation (3D conv + batch + channel)
    println!("\n🎯 Test 6: 5D convolution-like operations");
    
    // Simulate 3D convolution kernel [out_channels, in_channels, depth, height, width]
    let kernel_shape = vec![16, 3, 3, 3, 3]; // 16 output channels, 3 input channels, 3x3x3 kernel
    let kernel_data: Vec<f32> = (0..kernel_shape.iter().product::<usize>()).map(|i| (i as f32) * 0.01).collect();
    let kernel_5d = Tensor::from_vec(kernel_data, kernel_shape.clone(), &device)?;
    
    println!("5D kernel shape: {:?}", kernel_5d.shape());
    
    // Matrix multiplication along specific dimensions (simulating convolution)
    // For demonstration, we'll do element-wise ops since conv3d needs special implementation
    
    // Test 7: Advanced 5D operations
    println!("\n🔬 Test 7: Advanced 5D tensor operations");
    
    // Mean across different dimensions
    let mean_spatial = tensor_5d.mean(4)?; // Mean across width: [2, 3, 4, 8]
    let mean_batch = tensor_5d.mean(0)?;   // Mean across batch: [3, 4, 8, 8]
    
    println!("Mean across width: {:?}", mean_spatial.shape());
    println!("Mean across batch: {:?}", mean_batch.shape());
    
    // Sum across multiple dimensions
    let sum_spatial = tensor_5d.sum_keepdim((3, 4))?; // Sum across H,W: [2, 3, 4, 1, 1]
    println!("Sum across H,W: {:?}", sum_spatial.shape());
    
    // Test 8: 5D tensor indexing patterns
    println!("\n🎯 Test 8: Complex 5D indexing");
    
    // Create index tensors
    let indices = Tensor::arange(0u32, 4u32, &device)?; // [0, 1, 2, 3]
    
    // Advanced indexing examples
    let gathered = tensor_5d.gather(&indices.reshape((1, 1, 4, 1, 1))?, 2)?; // Gather along depth dimension
    println!("Gathered tensor shape: {:?}", gathered.shape());
    
    // Test 9: 5D tensor chunking and concatenation
    println!("\n🔗 Test 9: Chunking and concatenation");
    
    // Split along depth dimension
    let chunk_size = 2;
    let chunk1 = tensor_5d.narrow(2, 0, chunk_size)?; // [2, 3, 2, 8, 8]
    let chunk2 = tensor_5d.narrow(2, chunk_size, shape_5d[2] - chunk_size)?; // [2, 3, 2, 8, 8]
    
    println!("Chunk 1 shape: {:?}", chunk1.shape());
    println!("Chunk 2 shape: {:?}", chunk2.shape());
    
    // Concatenate back
    let concatenated = Tensor::cat(&[&chunk1, &chunk2], 2)?; // Should match original
    println!("Concatenated shape: {:?}", concatenated.shape());
    
    // Verify shapes match
    assert_eq!(tensor_5d.shape(), concatenated.shape());
    
    // Test 10: Memory and performance with 5D
    println!("\n⚡ Test 10: Performance characteristics");
    
    let start = std::time::Instant::now();
    
    // Create large 5D tensor
    let large_shape = vec![4, 32, 16, 64, 64]; // ~134M elements
    println!("Creating large 5D tensor: {:?}", large_shape);
    let large_tensor = Tensor::zeros(large_shape.clone(), DType::F32, &device)?;
    
    // Perform operations
    let result = (&large_tensor + 1.0)?.sqrt()?;
    
    let elapsed = start.elapsed();
    println!("Large 5D tensor ops completed in: {:.2}ms", elapsed.as_millis());
    println!("Final shape: {:?}", result.shape());
    
    println!("\n✅ All 5D tensor tests completed successfully!");
    println!("🎉 Candle fully supports 5D tensors with rich operations!");
    
    Ok(())
}

// Additional helper functions for 5D tensor operations
#[allow(dead_code)]
fn analyze_5d_tensor(tensor: &Tensor) -> Result<()> {
    println!("\n📊 5D Tensor Analysis:");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Rank: {}", tensor.rank());
    println!("  DType: {:?}", tensor.dtype());
    println!("  Device: {:?}", tensor.device());
    println!("  Element count: {}", tensor.elem_count());
    println!("  Memory usage: ~{:.2} MB", 
             tensor.elem_count() * tensor.dtype().size_in_bytes() / (1024 * 1024));
    
    Ok(())
}

#[allow(dead_code)]
fn create_video_like_tensor(batch: usize, channels: usize, frames: usize, height: usize, width: usize, device: &Device) -> Result<Tensor> {
    // Creates 5D tensor in video format [B, C, T, H, W]
    let shape = vec![batch, channels, frames, height, width];
    println!("Creating video-like 5D tensor: [B={}, C={}, T={}, H={}, W={}]", 
             batch, channels, frames, height, width);
    
    Tensor::zeros(shape.clone(), DType::F32, device)
}

#[allow(dead_code)]
fn create_3d_conv_weights(out_channels: usize, in_channels: usize, depth: usize, height: usize, width: usize, device: &Device) -> Result<Tensor> {
    // Creates 5D weight tensor for 3D convolution [O, I, D, H, W]
    let shape = vec![out_channels, in_channels, depth, height, width];
    println!("Creating 3D conv weights: [O={}, I={}, D={}, H={}, W={}]", 
             out_channels, in_channels, depth, height, width);
    
    // Initialize with Xavier/Glorot initialization
    let fan_in = in_channels * depth * height * width;
    let fan_out = out_channels * depth * height * width;
    let std = ((2.0 / (fan_in + fan_out) as f32)).sqrt();
    
    let data: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|_| fastrand::f32() * 2.0 * std - std)
        .collect();
    
    Tensor::from_vec(data, shape.clone(), device)
}
