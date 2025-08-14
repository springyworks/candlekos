//! Procedural field generation helpers for exploration demos.
//! These stay out of candle-core until they prove broadly useful.

use candle_core::{Device, Result, Tensor};

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
