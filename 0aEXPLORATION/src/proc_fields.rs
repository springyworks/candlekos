//! Procedural field generation helpers for exploration demos.
//! These stay out of candle-core until they prove broadly useful.

use candle_core::{Device, Result, Tensor};
// (No extra imports needed currently)

/// Create a 2D coordinate grid in normalized range [-1,1] along each axis.
/// Returns (grid_x, grid_y) each shaped [H, W].
pub fn meshgrid_2d(h: usize, w: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let xs: Vec<f32> = (0..w).map(|i| (i as f32 / (w - 1).max(1) as f32) * 2.0 - 1.0).collect();
    let ys: Vec<f32> = (0..h).map(|i| (i as f32 / (h - 1).max(1) as f32) * 2.0 - 1.0).collect();
    let x = Tensor::from_vec(xs, (w,), device)?;
    let y = Tensor::from_vec(ys, (h,), device)?;
    let grid_x = x.unsqueeze(0)?.repeat((h, 1))?; // [H,W]
    let grid_y = y.unsqueeze(1)?.repeat((1, w))?; // [H,W]
    Ok((grid_x, grid_y))
}

/// Radial distance field sqrt(x^2 + y^2) over normalized coords.
pub fn radial_field(h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let (gx, gy) = meshgrid_2d(h, w, device)?;
    (gx.sqr()? + gy.sqr()?)?.sqrt()
}

/// Generate a blended sinusoidal procedural texture in [0,1].
/// pattern = 0.5 + 0.5*sin(fr * r - t) + 0.25*sin(fx*x + fy*y + t)
pub fn sinusoidal_mix(
    h: usize,
    w: usize,
    time_scalar: f32,
    freq_radial: f64,
    freq_x: f64,
    freq_y: f64,
    device: &Device,
) -> Result<Tensor> {
    let r = radial_field(h, w, device)?;             // [H,W]
    let (gx, gy) = meshgrid_2d(h, w, device)?;       // [H,W]
    let t64 = time_scalar as f64;
    // wave1 = sin(fr * r - t)
    let wave1 = ((&r * freq_radial)? - t64)?.sin()?;
    let wave1 = ((wave1 * 0.5)? + 0.5f64)?;          // scale to [0,1]
    // wave2 = sin(fx*x + fy*y + t)
    let lin = ((&gx * freq_x)? + (&gy * freq_y)?)?;  // 5x + 3y style
    let wave2 = (lin + t64)?.sin()?;
    let wave2 = (wave2 * 0.25)?;                     // amplitude 0.25
    Ok((&wave1 + &wave2)?)                           // still roughly [0,1]
}

/// Generate a checkerboard pattern with given square size (in pixels). Values in [0,1].
pub fn checkerboard_field(h: usize, w: usize, square: usize, device: &Device) -> Result<Tensor> {
    let square = square.max(1);
    let mut vals = Vec::with_capacity(h * w);
    for y in 0..h { for x in 0..w { let v = ((x / square) + (y / square)) % 2; vals.push(v as f32); } }
    Tensor::from_vec(vals, (h, w), device)
}

/// Simple value noise (grid-based) with bilinear interpolation on a coarse lattice.
/// lattice_step: spacing between random anchor points.
pub fn value_noise_field(h: usize, w: usize, lattice_step: usize, device: &Device) -> Result<Tensor> {
    let step = lattice_step.max(2);
    let grid_h = (h + step - 1) / step + 1;
    let grid_w = (w + step - 1) / step + 1;
    let mut anchors = Vec::with_capacity(grid_h * grid_w);
    for _ in 0..(grid_h * grid_w) { anchors.push(fastrand::f32()); }
    let anchor_t = Tensor::from_vec(anchors, (grid_h, grid_w), device)?; // [Gh, Gw]
    // We'll compute noise on CPU directly without tensor ops for simplicity.
    let mut out = Vec::with_capacity(h * w);
    let anchors_vec = anchor_t.flatten_all()?.to_vec1::<f32>()?; // row-major
    let idx_anchor = |gy: usize, gx: usize| -> f32 { anchors_vec[gy * grid_w + gx] };
    for py in 0..h {
        for px in 0..w {
            let gx0 = (px / step).min(grid_w - 2);
            let gy0 = (py / step).min(grid_h - 2);
            let gx1 = gx0 + 1; let gy1 = gy0 + 1;
            let fx = (px as f32 % step as f32) / step as f32;
            let fy = (py as f32 % step as f32) / step as f32;
            let a00 = idx_anchor(gy0, gx0);
            let a10 = idx_anchor(gy0, gx1);
            let a01 = idx_anchor(gy1, gx0);
            let a11 = idx_anchor(gy1, gx1);
            let v0 = a00 * (1.0 - fx) + a10 * fx;
            let v1 = a01 * (1.0 - fx) + a11 * fx;
            let v = v0 * (1.0 - fy) + v1 * fy;
            out.push(v);
        }
    }
    Tensor::from_vec(out, (h, w), device)
}

/// Gaussian noise field in [0,1] produced by sampling N(0,1) and mapping via sigmoid.
pub fn gaussian_noise_field(h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let mut vals = Vec::with_capacity(h * w);
    // Box-Muller
    let mut i = 0;
    while i < h * w {
        let u1 = fastrand::f32().clamp(1e-6, 1.0);
        let u2 = fastrand::f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        vals.push(1.0 / (1.0 + (-z0).exp()));
        if i + 1 < h * w { vals.push(1.0 / (1.0 + (-z1).exp())); }
        i += 2;
    }
    Tensor::from_vec(vals, (h, w), device)
}

/// Generate a field from a user expression string using variables x,y (normalized [-1,1]) and t (time seconds).
/// Available functions depend on the `meval` crate (enable feature `expr-fields`).
#[cfg(feature = "expr-fields")]
pub fn expr_field(h: usize, w: usize, t: f32, expr: &str, device: &Device) -> Result<Tensor> {
    use meval::{Expr, Context};
    let parsed: Expr = expr.parse().map_err(|e| candle_core::Error::Msg(format!("expr parse error: {e}")))?;
    let (gx, gy) = meshgrid_2d(h, w, device)?; // [-1,1]
    let xv = gx.flatten_all()?.to_vec1::<f32>()?;
    let yv = gy.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(h * w);
    for (x,y) in xv.iter().zip(yv.iter()) {
        let mut ctx = Context::new();
        ctx.var("x", *x as f64);
        ctx.var("y", *y as f64);
        ctx.var("t", t as f64);
        let v = parsed.eval_with_context(ctx).map_err(|e| candle_core::Error::Msg(format!("eval error: {e}")))? as f32;
        // map roughly to [0,1] via 0.5+0.5*tanh
    let m: f32 = 0.5 + 0.5 * v.tanh();
    out.push(m.max(0.0).min(1.0));
    }
    Tensor::from_vec(out, (h, w), device)
}

/// Simple grayscale to RGBA u8 buffer (clamped) using provided colormap closure.
pub fn tensor_to_rgba<F>(t: &Tensor, colormap: F) -> Vec<u8>
where F: Fn(f32) -> (u8,u8,u8) {
    let Ok(v) = t.flatten_all().and_then(|f| f.to_vec1::<f32>()) else { return vec![]; };
    let dims = t.dims();
    let (h, w) = match dims {
        [h, w] => (*h, *w),
        _ => (1, v.len()),
    };
    let mut out = vec![0u8; h * w * 4];
    for i in 0..(h*w) {
        let mut x = v[i];
        if !x.is_finite() { x = 0.0; }
        x = x.clamp(0.0, 1.0);
        let (r,g,b) = colormap(x);
        out[i*4+0] = r;
        out[i*4+1] = g;
        out[i*4+2] = b;
        out[i*4+3] = 255;
    }
    out
}

/// Basic grayscale colormap.
pub fn gray(x: f32) -> (u8,u8,u8) {
    let c = (x * 255.0) as u8; (c,c,c)
}

/// Approximate plasma-style colormap.
pub fn plasma(x: f32) -> (u8,u8,u8) {
    let r = (0.5 + 1.5*x).sin()*0.5+0.5;
    let g = (0.5 + 1.5*(x+0.33)).sin()*0.5+0.5;
    let b = (0.5 + 1.5*(x+0.66)).sin()*0.5+0.5;
    ((r*255.0) as u8, (g*255.0) as u8, (b*255.0) as u8)
}
