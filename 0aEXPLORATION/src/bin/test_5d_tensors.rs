// 5D Tensor Operations Test
use candle_core::{Device, IndexOp, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let shape = [2, 3, 4, 4, 3];
    let data: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| i as f32 * 0.1)
        .collect();

    let tensor_5d = Tensor::from_vec(data, &shape, &device)?;
    println!("Created tensor shape: {:?}", tensor_5d.shape());

    let sum = tensor_5d.sum_all()?;
    println!("Sum: {:?}", sum.to_scalar::<f32>()?);

    let total_elements = shape.iter().product::<usize>();
    let reshaped = tensor_5d.reshape((2_usize, total_elements / 2))?;
    println!("Reshaped: {:?}", reshaped.shape());

    let first_batch = tensor_5d.i(0)?;
    println!("First batch: {:?}", first_batch.shape());

    println!("5D tensor operations completed!");
    Ok(())
}
