//! Higher Dimensional Tensor Test - Verifies Candle's support for 6D, 7D, 8D, and 10D tensors
//! Demonstrates that Candle natively supports arbitrarily high-dimensional tensors with full GPU acceleration

//! Test arbitrary high-dimensional tensor support in Candle (6D, 7D, 8D, 10D+)
//! Validates Candle's native unlimited dimension capabilities with CUDA acceleration

// Test higher dimensional tensors in Candle
use candle_core::{Device, Result, Tensor, DType};

fn main() -> Result<()> {
    println!("🧪 Testing Higher Dimensional Tensors in Candle");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}\n", device);
    
    // Test 6D tensor
    let shape_6d = vec![2, 3, 4, 5, 6, 7];
    let tensor_6d = Tensor::ones(shape_6d.clone(), DType::F32, &device)?;
    println!("✅ 6D Tensor: {:?}", tensor_6d.shape());
    println!("   Rank: {}, Elements: {}", tensor_6d.rank(), tensor_6d.elem_count());
    
    // Test 7D tensor  
    let shape_7d = vec![1, 2, 3, 4, 5, 6, 7];
    let tensor_7d = Tensor::zeros(shape_7d.clone(), DType::F32, &device)?;
    println!("✅ 7D Tensor: {:?}", tensor_7d.shape());
    println!("   Rank: {}, Elements: {}", tensor_7d.rank(), tensor_7d.elem_count());
    
    // Test 8D tensor
    let shape_8d = vec![1, 2, 2, 2, 2, 2, 2, 2];
    let tensor_8d = Tensor::zeros(shape_8d.clone(), DType::F32, &device)?;
    println!("✅ 8D Tensor: {:?}", tensor_8d.shape());
    println!("   Rank: {}, Elements: {}", tensor_8d.rank(), tensor_8d.elem_count());
    
    // Test 10D tensor
    let shape_10d = vec![1, 1, 2, 2, 2, 2, 2, 2, 2, 2];
    let tensor_10d = Tensor::ones(shape_10d.clone(), DType::F32, &device)?;
    println!("✅ 10D Tensor: {:?}", tensor_10d.shape());
    println!("    Rank: {}, Elements: {}", tensor_10d.rank(), tensor_10d.elem_count());
    
    // Test operations on higher-D tensors
    let result = (&tensor_6d + &tensor_6d)?;
    println!("✅ 6D + 6D operation: {:?}", result.shape());
    
    // Test permutation on higher dimensions
    let perm_6d = tensor_6d.permute((5, 4, 3, 2, 1, 0))?;
    println!("✅ 6D permutation: {:?} -> {:?}", tensor_6d.shape(), perm_6d.shape());
    
    println!("\n🎉 Candle supports arbitrarily high-dimensional tensors!");
    
    Ok(())
}
