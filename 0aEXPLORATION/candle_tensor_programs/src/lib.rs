//! candle_tensor_programs: exploration library for element-wise expression programs over tensors.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

pub mod lisp;

/// Contract
/// - Input: a tensor `t` and a program string `prog`
/// - Output: a tensor of same shape as `t`, computed element-wise
/// - Errors: parse or eval failure, dtype not supported
/// - Success: returns new tensor on same device
pub fn apply_program_str(t: &Tensor, prog: &str) -> Result<Tensor> {
    #[cfg(feature = "lisp")]
    {
        let expr = lisp::parse(prog).with_context(|| "parse lisp program")?;
        return apply_program_lisp(t, &expr);
    }
    #[cfg(not(feature = "lisp"))]
    {
        bail!("no evaluator features enabled; enable feature 'lisp'")
    }
}

#[cfg(feature = "lisp")]
pub fn apply_program_lisp(t: &Tensor, expr: &lisp::Expr) -> Result<Tensor> {
    // For now support f32 only to keep it simple; other dtypes can be added.
    if t.dtype() != DType::F32 {
        bail!("only f32 tensors are supported in this exploration crate")
    }
    // Compute element-wise by flattening -> host vec -> map -> from_vec -> reshape.
    let device = t.device();
    let shape = t.shape().clone();
    let flat = t.flatten_all()?;
    let data = flat.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(data.len());
    for x in data.into_iter() {
        let y = lisp::eval(expr, x)?;
        out.push(y);
    }
    let out = Tensor::from_vec(out, shape, device)?;
    Ok(out)
}
