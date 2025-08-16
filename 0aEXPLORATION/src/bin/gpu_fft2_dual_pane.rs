use candle_core::{Device, Result, Tensor};
use font8x8::legacy::BASIC_LEGACY as FONT8; // 8x8 ASCII bitmap font
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::time::Instant;

const WINDOW_WIDTH: usize = 1200;
const WINDOW_HEIGHT: usize = 600;
const TENSOR_W: usize = 256;
const TENSOR_H: usize = 256;
const PANE_W: usize = WINDOW_WIDTH / 2;

// Build normalized coordinate grid on the target device using tensor ops (stays on GPU).
fn grid(device: &Device, h: usize, w: usize, scale: f64) -> Result<(Tensor, Tensor)> {
    let xs = Tensor::arange(0f32, w as f32, device)?;
    let ys = Tensor::arange(0f32, h as f32, device)?;
    let grids = Tensor::meshgrid(&[&xs, &ys], true)?; // [X, Y], each HxW
    let mut x = grids[0].clone();
    let mut y = grids[1].clone();
    x = ((x - (w as f64 * 0.5))? / scale)?;
    y = ((y - (h as f64 * 0.5))? / scale)?;
    Ok((x, y))
}

// Hann window along both axes and zero-mean the image, all on device
fn preprocess_img(img: &Tensor) -> Result<Tensor> {
    let dims = img.dims();
    let (h, w) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let device = img.device();

    // Zero-mean (remove DC energy)
    let mean = img.mean_all()?;
    let centered = img.broadcast_sub(&mean)?;

    // 1D Hann windows
    let wy_idx = Tensor::arange(0f32, h as f32, &device)?;
    let wx_idx = Tensor::arange(0f32, w as f32, &device)?;
    let two_pi = std::f64::consts::PI * 2.0;
    let wy_rad = wy_idx.affine(two_pi / ((h.max(2) - 1) as f64), 0.0)?;
    let wx_rad = wx_idx.affine(two_pi / ((w.max(2) - 1) as f64), 0.0)?;
    let wy = wy_rad.cos()?.affine(-0.5, 0.5)?; // 0.5 - 0.5*cos
    let wx = wx_rad.cos()?.affine(-0.5, 0.5)?;
    let wy2 = wy.unsqueeze(1)?; // [H,1]
    let wx2 = wx.unsqueeze(0)?; // [1,W]
                                // Outer product to form 2D window: [H,1] x [1,W] -> [H,W]
    let window2d = wy2.matmul(&wx2)?;

    centered * &window2d
}

// Dramatic generator patterns, device-side
fn gen_pattern(device: &Device, which: usize, t: f32) -> Result<Tensor> {
    match which % 6 {
        // 0: Concentric breathing ripples
        0 => {
            let (x, y) = grid(device, TENSOR_H, TENSOR_W, 12.0)?;
            let r2 = x.sqr()?.add(&y.sqr()?)?;
            let r = r2.sqrt()?;
            let k = 2.6 + (t as f64 * 0.9).sin() * 0.7;
            let v = ((&r * k)? + (t as f64 * 2.0))?.sin()?;
            v.affine(0.5, 0.5)
        }
        // 1: Rotating Gabor with Gaussian envelope
        1 => {
            let (x, y) = grid(device, TENSOR_H, TENSOR_W, 1.0)?;
            let theta = (t as f64) * 1.3;
            let (ct, st) = (theta.cos(), theta.sin());
            let xr = (&x * ct)?.sub(&(&y * st)?)?;
            let yr = (&x * st)?.add(&(&y * ct)?)?;
            let carrier = ((&xr * 0.22)? + (t as f64 * 1.8))?.cos()?;
            let r2 = (&xr.sqr()? + &yr.sqr()?)?;
            let env = r2.affine(-1.0 / (2.0 * 40.0), 0.0)?.exp()?;
            (carrier * &env)?.affine(0.5, 0.5)
        }
        // 2: Moving Gaussian dots (two pulses crossing)
        2 => {
            let (x, y) = grid(device, TENSOR_H, TENSOR_W, 16.0)?;
            let cx = ((t * 1.8).sin() * 6.0) as f64;
            let cy = ((t * 1.1).cos() * 6.0) as f64;
            let dx1 = (&x - cx)?;
            let dy1 = (&y - cy)?;
            let dx2 = (&x + cx)?;
            let dy2 = (&y + cy)?;
            let r2a = (&dx1.sqr()? + &dy1.sqr()?)?;
            let r2b = (&dx2.sqr()? + &dy2.sqr()?)?;
            let a = r2a.affine(-1.0 / 4.0, 0.0)?.exp()?;
            let b = r2b.affine(-1.0 / 4.0, 0.0)?.exp()?;
            a.add(&b)?.affine(0.5, 0.5)
        }
        // 3: Expanding spiral chirp
        3 => {
            let (x, y) = grid(device, TENSOR_H, TENSOR_W, 14.0)?;
            let r2 = x.sqr()?.add(&y.sqr()?)?;
            let r = r2.sqrt()?;
            // Create a swirl-like term without atan2 by projecting on a rotating axis
            let theta = (t as f64) * 0.8;
            let swirl = (&x * theta.cos())?.add(&(&y * theta.sin())?)?;
            let f = 0.6 + 0.4 * (t as f64 * 0.7).cos();
            let a = (&r * f)?;
            let b = (&swirl * 0.7)?;
            let c = a.add(&b)?.affine(1.0, t as f64 * 1.4)?;
            let v = c.sin()?;
            v.affine(0.5, 0.5)
        }
        // 4: Lissajous checker-beats
        4 => {
            let (x, y) = grid(device, TENSOR_H, TENSOR_W, 1.0)?;
            let a = ((&x * 0.15)? + (t as f64 * 1.2))?.sin()?;
            let b = ((&y * 0.21)? + (t as f64 * 0.9 + 1.2))?.cos()?;
            (a * b)?.affine(0.5, 0.5)
        }
        // 5: Rotating square wave harmonics (sum of sines)
        _ => {
            let (x, y) = grid(device, TENSOR_H, TENSOR_W, 1.0)?;
            let th = (t as f64) * 0.5;
            let (ct, st) = (th.cos(), th.sin());
            let u = (&x * ct)?.sub(&(&y * st)?)?; // rotate
            let mut acc = Tensor::zeros(u.shape(), u.dtype(), device)?;
            // Add first few odd harmonics for a square-like wave along u
            for k in [1i32, 3, 5, 7] {
                let kf = k as f64;
                let term = ((&u * (0.1 * kf))?).sin()?;
                acc = acc.add(&term.affine(1.0 / kf, 0.0)?)?;
            }
            acc.affine(0.5, 0.5)
        }
    }
}

#[derive(Clone, Copy)]
struct VizOptions {
    gamma: f32,    // >0, e.g., 2.2; we apply powf(1/gamma)
    use_log: bool, // apply log1p mapping on normalized values
    log_k: f32,    // strength for log mapping
    fftshift: bool,
}

fn print_help(opts: &VizOptions, left_pat: usize, right_pat: usize, paused: bool) {
    let mode = if opts.use_log { "log" } else { "gamma" };
    println!(
        "\nControls:\n  Space: pause/resume (paused={})\n  1: gamma mapping\n  2: log mapping\n  Z/X: log strength (log_k={:.1})\n  -/=: gamma -/+ (gamma={:.2})\n  F: toggle fftshift (fftshift={})\n  A/S: left pattern prev/next (left={})\n  K/L: right pattern prev/next (right={})\n  H: show this help\n\nCurrent tone-map mode: {}\n",
        paused,
        opts.log_k,
        opts.gamma,
        opts.fftshift,
        left_pat,
        right_pat,
        mode
    );
}

// --- simple text drawing (8x8 bitmap) into ARGB buffer ---
fn blend_pixel(dst: &mut u32, src: u32) {
    // ARGB 8:8:8:8, src may contain alpha; do src over dst
    let sa = ((src >> 24) & 0xFF) as u32;
    if sa == 0 {
        return;
    }
    if sa == 255 {
        *dst = src;
        return;
    }
    let da = ((*dst >> 24) & 0xFF) as u32;
    let sr = ((src >> 16) & 0xFF) as u32;
    let sg = ((src >> 8) & 0xFF) as u32;
    let sb = (src & 0xFF) as u32;
    let dr = ((*dst >> 16) & 0xFF) as u32;
    let dg = ((*dst >> 8) & 0xFF) as u32;
    let db = (*dst & 0xFF) as u32;
    // pre-multiplied style blending: out = s + d*(1-a)
    let a = sa as f32 / 255.0;
    let na = (sa + ((da as f32) * (1.0 - a)) as u32).min(255);
    let rr = (sr as f32 * a + dr as f32 * (1.0 - a)).round() as u32;
    let rg = (sg as f32 * a + dg as f32 * (1.0 - a)).round() as u32;
    let rb = (sb as f32 * a + db as f32 * (1.0 - a)).round() as u32;
    *dst = (na << 24) | (rr << 16) | (rg << 8) | rb;
}

fn draw_char(
    buf: &mut [u32],
    stride: usize,
    x: usize,
    y: usize,
    ch: char,
    color: u32,
    scale: usize,
) {
    if ch as usize >= 128 {
        return;
    }
    let glyph = FONT8[ch as usize];
    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..8 {
            if (bits >> col) & 1 == 1 {
                // Draw left-to-right without horizontal flip
                let gx = x as isize + (col as isize) * (scale as isize);
                let gy = y as isize + (row as isize) * (scale as isize);
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = gx + dx as isize;
                        let py = gy + dy as isize;
                        if px >= 0 && py >= 0 {
                            let pxu = px as usize;
                            let pyu = py as usize;
                            if pxu < stride && pyu < WINDOW_HEIGHT {
                                let dst = &mut buf[pyu * stride + pxu];
                                blend_pixel(dst, color);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn draw_text(
    buf: &mut [u32],
    stride: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    scale: usize,
) {
    let mut cx = x;
    for ch in text.chars() {
        if ch == '\n' {
            cx = x;
            continue;
        }
        draw_char(buf, stride, cx, y, ch, color, scale);
        cx += 8 * scale + 1; // 1px spacing
        if cx >= stride {
            break;
        }
    }
}

fn text_size(text: &str, scale: usize) -> (usize, usize) {
    let w = text.chars().filter(|&c| c != '\n').count() * (8 * scale + 1);
    let h = 8 * scale;
    (w, h)
}

fn draw_rect(buf: &mut [u32], stride: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    for yy in y..y.saturating_add(h).min(WINDOW_HEIGHT) {
        for xx in x..x.saturating_add(w).min(stride) {
            let dst = &mut buf[yy * stride + xx];
            blend_pixel(dst, color);
        }
    }
}

fn pattern_name(idx: usize) -> &'static str {
    match idx % 6 {
        0 => "ripples",
        1 => "gabor",
        2 => "cross-gauss",
        3 => "spiral",
        4 => "lissajous",
        _ => "square-harm",
    }
}

fn tensor_to_pixels(img: &Tensor, x_off: usize, opts: VizOptions) -> Result<Vec<u32>> {
    let cpu = img.to_device(&Device::Cpu)?;
    let dims = cpu.dims().to_vec();
    let (h, w) = if dims.len() >= 2 {
        (dims[dims.len() - 2], dims[dims.len() - 1])
    } else if dims.len() == 1 {
        (1, dims[0])
    } else {
        (1, 1)
    };

    let flat = cpu.flatten_all()?;
    let v = flat.to_vec1::<f32>()?;
    let mut px = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];
    let sx = PANE_W as f32 / w as f32;
    let sy = WINDOW_HEIGHT as f32 / h as f32;
    for y in 0..WINDOW_HEIGHT {
        for x in 0..PANE_W {
            let tx = (x as f32 / sx) as usize;
            let ty = (y as f32 / sy) as usize;
            if tx < w && ty < h {
                // Optional fftshift: center the DC component
                let (txs, tys) = if opts.fftshift {
                    ((tx + w / 2) % w, (ty + h / 2) % h)
                } else {
                    (tx, ty)
                };
                let mut val = v[tys * w + txs].clamp(0.0, 1.0);
                // Tone map: log or gamma on normalized magnitude
                if opts.use_log {
                    let k = opts.log_k.max(1e-6);
                    val = ((1.0 + k * val).ln()) / (1.0 + k).ln();
                } else {
                    let g = opts.gamma.max(1e-6);
                    val = val.powf(1.0 / g);
                }
                let c = (val * 255.0) as u8;
                let color = 0xFF000000 | ((c as u32) << 16) | ((c as u32) << 8) | c as u32;
                let xx = x + x_off;
                if xx < WINDOW_WIDTH {
                    px[y * WINDOW_WIDTH + xx] = color;
                }
            }
        }
    }
    Ok(px)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device.location());

    let mut t: f32 = 0.0;
    let mut left_pat: usize = 1; // start with Gabor (good FFT)
    let mut right_pat: usize = 3; // spiral chirp
    let mut paused = false;
    let mut opts = VizOptions {
        gamma: 2.2,
        use_log: true,
        log_k: 60.0,
        fftshift: true,
    };
    let mut last = Instant::now();

    let mut win = Window::new(
        "GPU 2D FFT (VkFFT) - Dual Pane",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    // Show legend once at startup
    print_help(&opts, left_pat, right_pat, paused);

    while win.is_open() && !win.is_key_down(Key::Escape) {
        // Time step
        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        if !paused {
            t += (dt * 60.0).min(1.0) * 0.03; // speed factor, stable even if dt spikes
        }

        // Controls
        for k in win.get_keys_pressed(KeyRepeat::Yes) {
            match k {
                Key::Space => {
                    paused = !paused;
                    println!("paused: {}", paused);
                }
                Key::Key1 => {
                    opts.use_log = false;
                    println!("tone map: gamma (gamma={:.2})", opts.gamma);
                }
                Key::Key2 => {
                    opts.use_log = true;
                    println!("tone map: log (log_k={:.1})", opts.log_k);
                }
                Key::F => {
                    opts.fftshift = !opts.fftshift;
                    println!("fftshift: {}", opts.fftshift);
                }
                Key::Equal => {
                    opts.gamma = (opts.gamma + 0.2).min(6.0);
                    println!("gamma -> {:.2}", opts.gamma);
                } // '+'
                Key::Minus => {
                    opts.gamma = (opts.gamma - 0.2).max(0.2);
                    println!("gamma -> {:.2}", opts.gamma);
                }
                // Z/X change log strength on builds where bracket keys aren't available in minifb
                Key::Z => {
                    opts.log_k = (opts.log_k - 2.0).max(1.0);
                    println!("log_k -> {:.1}", opts.log_k);
                }
                Key::X => {
                    opts.log_k = (opts.log_k + 2.0).min(1000.0);
                    println!("log_k -> {:.1}", opts.log_k);
                }
                // Pattern selection
                Key::A => {
                    left_pat = (left_pat + 5) % 6;
                    println!("left pattern: {}", left_pat);
                }
                Key::S => {
                    left_pat = (left_pat + 1) % 6;
                    println!("left pattern: {}", left_pat);
                }
                Key::K => {
                    right_pat = (right_pat + 5) % 6;
                    println!("right pattern: {}", right_pat);
                }
                Key::L => {
                    right_pat = (right_pat + 1) % 6;
                    println!("right pattern: {}", right_pat);
                }
                Key::H => {
                    print_help(&opts, left_pat, right_pat, paused);
                }
                _ => {}
            }
        }

        // Generate animated inputs (pure tensor math on device) + preprocessing to kill DC
        let img_a = preprocess_img(&gen_pattern(&device, left_pat, t)?)?;
        let img_b = preprocess_img(&gen_pattern(&device, right_pat, t * 0.9)?)?;

        // FFT -> magnitude -> per-frame normalization to [0,1]
        let spec_a = img_a.fft2(true, true)?;
        let spec_b = img_b.fft2(true, true)?;
        let mag_a = spec_a.fft_magnitude()?;
        let mag_b = spec_b.fft_magnitude()?;
        let max_a = mag_a.max_all()?.to_scalar::<f32>()?;
        let max_b = mag_b.max_all()?.to_scalar::<f32>()?;
        let mag_a = mag_a.affine(1.0f64 / (max_a as f64 + 1e-6), 0.0)?;
        let mag_b = mag_b.affine(1.0f64 / (max_b as f64 + 1e-6), 0.0)?;

        // Rasterize
        let left = tensor_to_pixels(&mag_a, 0, opts)?;
        let right = tensor_to_pixels(&mag_b, PANE_W, opts)?;
        let mut buf = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];
        for i in 0..buf.len() {
            buf[i] = left[i] | right[i];
        }
        // separator
        for y in 0..WINDOW_HEIGHT {
            buf[y * WINDOW_WIDTH + PANE_W] = 0xFF00FF00;
        }

        // Overlays: compact status text per pane
        let scale = 2usize; // larger for readability
        let white = 0xFFFFFFFF;
        let shadow = 0x80000000; // semi-transparent black
        let bg = 0x7F000000; // translucent dark background
                             // Left title
        let left_title = format!("L | {}", pattern_name(left_pat));
        let (ltw, lth) = text_size(&left_title, scale);
        draw_rect(&mut buf, WINDOW_WIDTH, 6, 6, ltw + 6, lth + 6, bg);
        draw_text(&mut buf, WINDOW_WIDTH, 9, 9, &left_title, shadow, scale);
        draw_text(&mut buf, WINDOW_WIDTH, 8, 8, &left_title, white, scale);
        // Right title (offset by pane)
        let right_title = format!("R | {}", pattern_name(right_pat));
        let (rtw, rth) = text_size(&right_title, scale);
        draw_rect(&mut buf, WINDOW_WIDTH, PANE_W + 6, 6, rtw + 6, rth + 6, bg);
        draw_text(
            &mut buf,
            WINDOW_WIDTH,
            PANE_W + 9,
            9,
            &right_title,
            shadow,
            scale,
        );
        draw_text(
            &mut buf,
            WINDOW_WIDTH,
            PANE_W + 8,
            8,
            &right_title,
            white,
            scale,
        );
        // Global status at bottom-left
        let status = if opts.use_log {
            format!(
                "mode=log(k={:.0}) shift={} paused={}",
                opts.log_k, opts.fftshift, paused
            )
        } else {
            format!(
                "mode=gamma(g={:.1}) shift={} paused={}",
                opts.gamma, opts.fftshift, paused
            )
        };
        let (sw, sh) = text_size(&status, scale);
        let sy0 = WINDOW_HEIGHT.saturating_sub(sh + 10);
        draw_rect(&mut buf, WINDOW_WIDTH, 6, sy0 - 2, sw + 6, sh + 6, bg);
        draw_text(&mut buf, WINDOW_WIDTH, 9, sy0 + 1, &status, shadow, scale);
        draw_text(&mut buf, WINDOW_WIDTH, 8, sy0, &status, white, scale);
        win.update_with_buffer(&buf, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }

    Ok(())
}
