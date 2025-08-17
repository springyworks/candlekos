#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn lisp_square_plus_sin() {
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.5], 4, &dev).unwrap();
        let out = apply_program_str(&t, "(+ (sqr x) (sin x))").unwrap();
        let v = out.to_vec1::<f32>().unwrap();
        let expected: Vec<f32> = [0.0f32, 1.0, 4.0, 12.25]
            .into_iter()
            .zip([0.0f32, 1.0f32.sin(), 2.0f32.sin(), 3.5f32.sin()])
            .map(|(a, b)| a + b)
            .collect();
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }
}
