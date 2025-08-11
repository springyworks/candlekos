//! Custom Types Investigation - Analysis of Candle's tensor element type limitations and requirements
//! Explores why custom compound types aren't supported and demonstrates workaround approaches for complex data

//! Analysis of custom type limitations in Candle tensors and available workarounds
//! Explores why compound types aren't supported and demonstrates practical alternatives

// Test custom types in Candle tensors
use candle_core::{Device, Result, Tensor, DType};

// Let's try to understand what custom types are possible
fn main() -> Result<()> {
    println!("🧪 Testing Custom Types in Candle Tensors");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}\n", device);
    
    // Current supported types in Candle
    println!("📋 Currently supported types in Candle:");
    println!("  ✅ u8, u32, i64 (integers)");
    println!("  ✅ f16, bf16, f32, f64 (floats)");
    println!("  ✅ F8E4M3 (8-bit float)\n");
    
    // Try with basic types
    let tensor_u8 = Tensor::ones((2, 3), DType::U8, &device)?;
    let tensor_f32 = Tensor::ones((2, 3), DType::F32, &device)?;
    
    println!("✅ Basic types work: u8 tensor {:?}", tensor_u8.shape());
    println!("✅ Basic types work: f32 tensor {:?}", tensor_f32.shape());
    
    // What about complex numbers? Let's see what we'd need
    println!("\n🔍 Analysis of custom type requirements:");
    println!("  The WithDType trait requires:");
    println!("    - Copy + Sized");
    println!("    - num_traits::NumAssign");
    println!("    - PartialOrd + Display");
    println!("    - Send + Sync + 'static");
    println!("    - VecOps for CPU operations");
    println!("    - Integration with Storage system");
    
    println!("\n❌ Custom compound types are NOT currently supported because:");
    println!("  1. DType enum is closed - only 8 built-in types");
    println!("  2. Storage system (CpuStorage, CudaStorage) is closed");
    println!("  3. All backend operations are hardcoded for built-in types");
    println!("  4. GPU kernels only exist for supported types");
    
    println!("\n💡 Workarounds for compound data:");
    
    // Workaround 1: Multiple tensors
    println!("  Approach 1: Multiple tensors for compound data");
    let real_part = Tensor::ones((2, 3), DType::F32, &device)?;
    let imag_part = Tensor::zeros((2, 3), DType::F32, &device)?;
    println!("    Complex number as: real_tensor + i*imag_tensor");
    println!("    Real: {:?}, Imag: {:?}", real_part.shape(), imag_part.shape());
    
    // Workaround 2: Packed representation
    println!("  Approach 2: Packed representation in existing types");
    let complex_packed = Tensor::zeros((2, 3, 2), DType::F32, &device)?; // [..., 0]=real, [..., 1]=imag
    println!("    Complex packed: {:?} where last dim [real, imag]", complex_packed.shape());
    
    // Workaround 3: Interleaved
    println!("  Approach 3: Interleaved data");
    let interleaved = Tensor::zeros((2, 6), DType::F32, &device)?; // [r1,i1,r2,i2,r3,i3]
    println!("    Interleaved: {:?} where data is [r,i,r,i,...]", interleaved.shape());
    
    println!("\n🎯 Conclusion:");
    println!("  Candle tensors are LIMITED to built-in scalar types only.");
    println!("  No custom structs, enums, or compound types are supported.");
    println!("  This is by design for GPU compatibility and performance.");
    
    Ok(())
}
