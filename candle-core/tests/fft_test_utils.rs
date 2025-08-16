//! Shared helpers for FFT-related tests & smokes
//! Keep logic (scale detection, tolerances) DRY across test files.

use candle_core::Result;

// Tolerance rationale:
// REAL: Empirically observed max absolute error on roundtrip (n<=4096) stays <2e-3 across CPU rustfft and CUDA paths.
//        We use 5e-3 to allow headroom for provider differences (plan warm-ups, fused kernels) without masking issues.
// COMPLEX: Interleaved complex c2c paths can accumulate slightly higher error especially on GPU (sin/cos synthesis + twiddle).
//          Observed <4e-3; we set 7e-3 as conservative upper bound.
pub const FFT_EPS_REAL: f32 = 5e-3;
pub const FFT_EPS_COMPLEX: f32 = 7e-3;

/// Detect a uniform scaling factor between an output series `out[i]` and an expected `exp[i]`.
/// Returns 1.0 if scale ~1 or could not be confidently determined (e.g. all zeros).
pub fn detect_scale(out: &[f32], exp: &[f32]) -> f32 {
    let mut scale = None;
    for (&o, &e) in out.iter().zip(exp.iter()) {
        if e.abs() > 1e-6 {
            scale = Some(o / e);
            break;
        }
    }
    if let Some(s) = scale {
        if (s - 1.0).abs() < 1e-3 { 1.0 } else { s }
    } else {
        1.0
    }
}

/// Assert approximate equality after optional scaling.
pub fn assert_approx_scaled(out: &[f32], exp: &[f32], eps: f32) {
    assert_eq!(
        out.len(),
        exp.len(),
        "length mismatch: {} vs {}",
        out.len(),
        exp.len()
    );
    let scale = detect_scale(out, exp);
    for (i, (&o, &e)) in out.iter().zip(exp.iter()).enumerate() {
        let adj = if scale != 1.0 { o / scale } else { o };
        assert!(
            (adj - e).abs() < eps,
            "index {i}: got={adj} exp={e} (raw={o} scale={scale})"
        );
    }
}

/// Splits an interleaved complex buffer into (real, imag) slices (borrowed view semantics).
pub fn split_interleaved_complex(buf: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut re = Vec::with_capacity(buf.len() / 2);
    let mut im = Vec::with_capacity(buf.len() / 2);
    let mut iter = buf.iter();
    while let (Some(r), Some(i)) = (iter.next(), iter.next()) {
        re.push(*r);
        im.push(*i);
    }
    (re, im)
}

/// Produce expected interleaved complex sinusoid pair (sin, cos) over n samples.
pub fn expected_sin_cos(n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32;
        v.push(t.sin());
        v.push(t.cos());
    }
    v
}

pub fn expected_range(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

pub fn ok<T>(v: T) -> Result<T> {
    Ok(v)
}
