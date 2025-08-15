use candle_core::{Device, Tensor};
use meval::{Context, Expr};
use std::str::FromStr;

pub trait TensorAugment {
    /// Example: Compute the sum of all elements in the tensor (f32 only)
    fn sum_elements(&self) -> candle_core::Result<f32>;
}

impl TensorAugment for Tensor {
    fn sum_elements(&self) -> candle_core::Result<f32> {
        let v = self.to_vec1::<f32>()?;
        Ok(v.iter().copied().sum())
    }
}

/// Trait to fill a tensor with values from a math expression string.
pub trait TensorMathFill {
    /// Fill a tensor of shape (h, w) with values from a math expression string.
    /// The expression can use variables x and y (normalized to [0,1]).
    fn fill_with_expr(h: usize, w: usize, expr: &str, device: &Device) -> candle_core::Result<Tensor>;
}

impl TensorMathFill for Tensor {
    fn fill_with_expr(h: usize, w: usize, expr: &str, device: &Device) -> candle_core::Result<Tensor> {
        let parsed = Expr::from_str(expr).map_err(|e| candle_core::Error::Msg(format!("Parse error: {}", e)))?;
        let mut data = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let x_norm = x as f64 / (w as f64 - 1.0);
                let y_norm = y as f64 / (h as f64 - 1.0);
                let mut ctx = Context::new();
                // Variables
                ctx.var("x", x_norm);
                ctx.var("y", y_norm);
                // Common math functions (extend as needed)
                ctx.func("sin", |v: f64| v.sin());
                ctx.func("cos", |v: f64| v.cos());
                ctx.func("tan", |v: f64| v.tan());
                ctx.func("sqrt", |v: f64| v.sqrt());
                ctx.func("abs", |v: f64| v.abs());
                let val = parsed
                    .eval_with_context(ctx)
                    .map_err(|e| candle_core::Error::Msg(format!("Eval error: {}", e)))?;
                data.push(val as f32);
            }
        }
        Tensor::from_vec(data, (h, w), device)
    }
}
// You can add more trait methods here to extend Tensor functionality.
