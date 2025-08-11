use candle::{Tensor, Device, Result};

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;    
    // Smoke test: single-row contiguous scan
    let t0 = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &dev)?;
    match t0.inclusive_scan(1) {
        Ok(res0) => println!("smoke contiguous inclusive: {:?}", res0.to_vec2::<f32>()?),
        Err(e) => { println!("smoke contiguous error: {:?}", e); return Ok(()); }
    }
    // Row-wise scan on 2x4 tensor
    let t = Tensor::from_vec(vec![1f32,2.,3.,4.,5.,6.,7.,8.], (2,4), &dev)?;
    let inc = match t.inclusive_scan(1) {
        Ok(v) => v,
        Err(e) => { println!("inclusive_scan error: {:?}", e); return Ok(()); }
    };
    let exc = match t.exclusive_scan(1) {
        Ok(v) => v,
        Err(e) => { println!("exclusive_scan error: {:?}", e); return Ok(()); }
    };
    println!("input: {:?}", t.to_vec2::<f32>()?);
    println!("inclusive: {:?}", inc.to_vec2::<f32>()?);
    println!("exclusive: {:?}", exc.to_vec2::<f32>()?);
    Ok(())
}
