#[cfg(test)]
mod cpu_scan_investigation {
    use candle_core::{Tensor, Device, Result};

    #[test]
    fn test_cpu_scan_methods() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
        
        println!("Testing CPU scan operations:");
        println!("Input: {:?}", input.to_vec1::<f32>()?);
        
        // Test cumsum (matrix multiplication approach)
        println!("\n1. Testing cumsum (matrix multiplication approach):");
        let cumsum_result = input.cumsum(0)?;
        println!("   cumsum result: {:?}", cumsum_result.to_vec1::<f32>()?);
        
        // Test inclusive_scan (should also use cumsum fallback on CPU)
        println!("\n2. Testing inclusive_scan (should fallback to cumsum):");
        let inclusive_result = input.inclusive_scan(0)?;
        println!("   inclusive_scan result: {:?}", inclusive_result.to_vec1::<f32>()?);
        
        // Test exclusive_scan (should also use cumsum-based approach on CPU)
        println!("\n3. Testing exclusive_scan (should use cumsum-based approach):");
        let exclusive_result = input.exclusive_scan(0)?;
        println!("   exclusive_scan result: {:?}", exclusive_result.to_vec1::<f32>()?);
        
        // Verify results are correct
        assert_eq!(cumsum_result.to_vec1::<f32>()?, &[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(inclusive_result.to_vec1::<f32>()?, &[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(exclusive_result.to_vec1::<f32>()?, &[0.0, 1.0, 3.0, 6.0]);
        
        Ok(())
    }
}
