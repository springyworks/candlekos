use candle_core::{Tensor, Device, Result};

// Retained as an ignored exploratory test; trimmed verbosity to keep logs clean.
#[test]
#[ignore]
fn cpu_scan_investigation() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let cumsum_result = input.cumsum(0)?;
    let inclusive_result = input.inclusive_scan(0)?;
    let exclusive_result = input.exclusive_scan(0)?;
    assert_eq!(cumsum_result.to_vec1::<f32>()?, &[1.0, 3.0, 6.0, 10.0]);
    assert_eq!(inclusive_result.to_vec1::<f32>()?, &[1.0, 3.0, 6.0, 10.0]);
    assert_eq!(exclusive_result.to_vec1::<f32>()?, &[0.0, 1.0, 3.0, 6.0]);
    Ok(())
}
