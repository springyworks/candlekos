pub use candle_core as candle;
pub use candle_nn as nn;
pub use image;
pub use base64;
// Do not re-export anyhow to avoid name conflicts in evcxr.

use ::base64::Engine as _;
use ::base64::engine::general_purpose::STANDARD as BASE64;
use ::image::ImageEncoder;
use ::anyhow::Result;
use candle_core::{Tensor, Device};
use std::sync::{Mutex, OnceLock};
use std::path::{PathBuf, Path};
use std::fs;
use std::fmt::Write as _;

/// Display a single-channel f32 tensor as grayscale PNG in evcxr output.
/// - Expects shape (H, W) or (1, H, W). Values outside [0,1] will be clamped.
pub fn show_tensor_gray(img: &Tensor) -> Result<()> {
	let img = if img.dims().len() == 3 { img.squeeze(0)? } else { img.clone() };
	let dims = img.dims();
	if dims.len() != 2 { ::anyhow::bail!("show_tensor_gray expects (H,W) or (1,H,W), got {:?}", dims); }
	let (h, w) = (dims[0], dims[1]);
	let cpu = img.to_device(&Device::Cpu)?;
	let v = cpu.to_vec2::<f32>()?;
	// Convert to u8, clamp [0,255]
	let mut buf = vec![0u8; h * w];
	for y in 0..h {
		for x in 0..w {
			let mut p = v[y][x];
			if p < 0.0 { p = 0.0; }
			if p > 1.0 { p = 1.0; }
			buf[y * w + x] = (p * 255.0).round() as u8;
		}
	}
	let img = image::GrayImage::from_raw(w as u32, h as u32, buf)
		.ok_or_else(|| ::anyhow::anyhow!("failed to build gray image"))?;
	let mut png = Vec::new();
	let enc = image::codecs::png::PngEncoder::new(&mut png);
	enc.write_image(
		img.as_raw(),
		w as u32,
		h as u32,
		image::ColorType::L8,
	)?;
	// Save to disk if configured.
	let _ = save_png_if_configured(&png, "gray");
	let b64 = BASE64.encode(png);
	println!(
		"EVCXR_BEGIN_CONTENT image/png\n{}\nEVCXR_END_CONTENT",
		b64
	);
	Ok(())
}

/// Display an RGB f32 tensor as color PNG in evcxr output.
/// - Expects shape (3, H, W). Values in [0,1] recommended; clamped otherwise.
pub fn show_tensor_rgb(img: &Tensor) -> Result<()> {
	let dims = img.dims();
	if dims.len() != 3 || dims[0] != 3 {
		::anyhow::bail!("show_tensor_rgb expects (3,H,W), got {:?}", dims);
	}
	let (h, w) = (dims[1], dims[2]);
	let cpu = img.to_device(&Device::Cpu)?;
	let v = cpu.to_vec3::<f32>()?; // [3][H][W]
	let mut buf = vec![0u8; h * w * 3];
	for y in 0..h {
		for x in 0..w {
			let r = (v[0][y][x].clamp(0.0, 1.0) * 255.0).round() as u8;
			let g = (v[1][y][x].clamp(0.0, 1.0) * 255.0).round() as u8;
			let b = (v[2][y][x].clamp(0.0, 1.0) * 255.0).round() as u8;
			let idx = (y * w + x) * 3;
			buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b;
		}
	}
	let img = image::RgbImage::from_raw(w as u32, h as u32, buf)
		.ok_or_else(|| ::anyhow::anyhow!("failed to build rgb image"))?;
	let mut png = Vec::new();
	let enc = image::codecs::png::PngEncoder::new(&mut png);
	enc.write_image(
		img.as_raw(),
		w as u32,
		h as u32,
		image::ColorType::Rgb8,
	)?;
	// Save to disk if configured.
	let _ = save_png_if_configured(&png, "rgb");
	let b64 = BASE64.encode(png);
	println!(
		"EVCXR_BEGIN_CONTENT image/png\n{}\nEVCXR_END_CONTENT",
		b64
	);
	Ok(())
}

/// Return a data URL (image/png;base64,...) for an RGB tensor (3,H,W) in [0,1].
pub fn tensor_to_png_data_url_rgb(img: &Tensor) -> Result<String> {
	let dims = img.dims();
	if dims.len() != 3 || dims[0] != 3 {
		::anyhow::bail!("tensor_to_png_data_url_rgb expects (3,H,W), got {:?}", dims);
	}
	let (h, w) = (dims[1], dims[2]);
	let cpu = img.to_device(&Device::Cpu)?;
	let v = cpu.to_vec3::<f32>()?;
	let mut buf = vec![0u8; h*w*3];
	for y in 0..h { for x in 0..w {
		let i = (y*w + x)*3;
		buf[i+0] = (v[0][y][x].clamp(0.0,1.0)*255.0).round() as u8;
		buf[i+1] = (v[1][y][x].clamp(0.0,1.0)*255.0).round() as u8;
		buf[i+2] = (v[2][y][x].clamp(0.0,1.0)*255.0).round() as u8;
	}}
    let img = image::RgbImage::from_raw(w as u32, h as u32, buf)
		.ok_or_else(|| ::anyhow::anyhow!("failed to build rgb image"))?;
	let mut png = Vec::new();
	let enc = image::codecs::png::PngEncoder::new(&mut png);
	enc.write_image(img.as_raw(), w as u32, h as u32, image::ColorType::Rgb8)?;
	Ok(format!("data:image/png;base64,{}", BASE64.encode(png)))
}

/// Return a data URL (image/png;base64,...) for a grayscale tensor (H,W) or (1,H,W) in [0,1].
pub fn tensor_to_png_data_url_gray(img: &Tensor) -> Result<String> {
	let img = if img.dims().len() == 3 { img.squeeze(0)? } else { img.clone() };
	let dims = img.dims();
	if dims.len() != 2 { ::anyhow::bail!("tensor_to_png_data_url_gray expects (H,W) or (1,H,W), got {:?}", dims); }
	let (h, w) = (dims[0], dims[1]);
	let cpu = img.to_device(&Device::Cpu)?;
	let v = cpu.to_vec2::<f32>()?;
	let mut buf = vec![0u8; h * w];
	for y in 0..h { for x in 0..w {
		let mut p = v[y][x];
		if p < 0.0 { p = 0.0; }
		if p > 1.0 { p = 1.0; }
		buf[y * w + x] = (p * 255.0).round() as u8;
	}}
	let img = image::GrayImage::from_raw(w as u32, h as u32, buf)
		.ok_or_else(|| ::anyhow::anyhow!("failed to build gray image"))?;
	let mut png = Vec::new();
	let enc = image::codecs::png::PngEncoder::new(&mut png);
	enc.write_image(img.as_raw(), w as u32, h as u32, image::ColorType::L8)?;
	Ok(format!("data:image/png;base64,{}", BASE64.encode(png)))
}

// ----------------------------
// Optional image store support
// ----------------------------

static IMAGE_STORE: OnceLock<Mutex<(Option<PathBuf>, u64)>> = OnceLock::new();
static LAST_SAVED: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn image_store() -> &'static Mutex<(Option<PathBuf>, u64)> {
	IMAGE_STORE.get_or_init(|| Mutex::new((None, 0)))
}

fn last_saved_path() -> &'static Mutex<Option<PathBuf>> {
	LAST_SAVED.get_or_init(|| Mutex::new(None))
}

/// Set a relative directory where helper PNGs will be saved. The directory will be created if missing.
/// The path must be relative to the current working directory of the kernel.
pub fn set_image_store_rel_dir(dir: &str) -> Result<()> {
	let p = Path::new(dir);
	if p.is_absolute() {
		::anyhow::bail!("set_image_store_rel_dir expects a relative path, got: {}", dir);
	}
	fs::create_dir_all(p)?;
	let mut guard = image_store().lock().expect("image_store lock");
	guard.0 = Some(p.to_path_buf());
	guard.1 = 0; // reset counter when setting a new dir
	drop(guard);
	// Try to resolve absolute path for user visibility.
	let abs = match std::fs::canonicalize(p) {
		Ok(ap) => ap,
		Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(p),
	};
	let html = format!(
		"<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 11px; color: #5e6a75;\">Saving images to: <code>{}</code></div>",
		abs.display()
	);
	println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", html);
	Ok(())
}


fn save_png_if_configured(png: &[u8], prefix: &str) -> Option<PathBuf> {
	let mut guard = image_store().lock().ok()?;
	let dir = guard.0.as_ref()?.clone();
	// Generate a simple incremental filename.
	guard.1 += 1;
	let n = guard.1;
	drop(guard); // release lock before filesystem write
	let filename = format!("{}_{:04}.png", prefix, n);
	let path = dir.join(filename);
	if let Err(e) = fs::write(&path, png) {
		eprintln!("warn: failed to save PNG to {}: {}", path.display(), e);
		return None;
	}
	if let Ok(mut last) = last_saved_path().lock() { *last = Some(path.clone()); }
	Some(path)
}

/// Escape minimal HTML entities for safe caption rendering.
fn escape_html(s: &str) -> String {
	s.replace('&', "&amp;")
		.replace('<', "&lt;")
		.replace('>', "&gt;")
}

/// Helper to print a thin, monospace HTML caption block under an image in evcxr.
fn caption_html_string(shape: &str, caption: Option<&str>, input_desc: Option<&str>, output_desc: Option<&str>, saved: Option<&Path>) -> String {
	let mut body = String::new();
	if let Some(cap) = caption {
		if !cap.trim().is_empty() {
			let _ = writeln!(body, "{}", escape_html(cap));
		}
	}
	let _ = writeln!(body, "Dims: {}", escape_html(shape));
	if let Some(inp) = input_desc {
		if !inp.trim().is_empty() {
			let _ = writeln!(body, "Input: {}", escape_html(inp));
		}
	}
	if let Some(out) = output_desc {
		if !out.trim().is_empty() {
			let _ = writeln!(body, "Output: {}", escape_html(out));
		}
	}
	if let Some(p) = saved {
		let _ = writeln!(body, "Saved: {}", escape_html(&p.display().to_string()));
	}
	format!(
		"<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px; border-top: 1px solid rgba(127,127,127,0.35); margin-top: 6px; padding-top: 6px; white-space: pre; line-height: 1.3; letter-spacing: 0.2px;\">\n{}\n</div>",
		body
	)
}

/// Display grayscale image then print a caption under it (HTML, monospace).
/// - Supports (H,W) or (1,H,W). Values are clamped to [0,1].
/// - The PNG is emitted first (so VS Code Plots captures it), then an HTML caption
///   is emitted with dims and optional caption/input/output descriptions.
pub fn show_tensor_gray_captioned(
	img: &Tensor,
	caption: Option<&str>,
	input_desc: Option<&str>,
	output_desc: Option<&str>,
) -> Result<()> {
	// Emit the image as PNG first for the Plot viewer.
	show_tensor_gray(img)?;
	// Build dims string from the original shape.
	let dims = img.dims();
	let shape = if dims.is_empty() {
		"(scalar)".to_string()
	} else {
		dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×")
	};
	// Grab last-saved image path if any, and emit a single HTML block containing the inline <img> and the caption.
	let saved = last_saved_path().lock().ok().and_then(|g| g.clone());
	let img_html = match tensor_to_png_data_url_gray(img) {
		Ok(url) => format!(
			"<div><img style=\"max-width:100%; image-rendering: pixelated; border: 1px solid rgba(0,0,0,0.12);\" src=\"{}\"/></div>",
			url
		),
		Err(_) => String::new(),
	};
	let cap_html = caption_html_string(&shape, caption, input_desc, output_desc, saved.as_deref());
	let combined = format!("{}\n{}", img_html, cap_html);
	println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", combined);
	// Plain-text fallback for terminals/viewers that hide HTML captions
	let mut plain = String::new();
	if let Some(c) = caption { if !c.is_empty() { let _ = write!(plain, "Caption: {}  ", c); } }
	let _ = write!(plain, "Dims: {}", shape);
	if let Some(p) = saved.as_deref() { let _ = write!(plain, "  Saved: {}", p.display()); }
	println!("{}", plain);
	Ok(())
}

/// Display RGB image then print a caption under it (HTML, monospace).
/// - Expects (3,H,W). Values are clamped to [0,1].
/// - The PNG is emitted first (so VS Code Plots captures it), then an HTML caption
///   is emitted with dims and optional caption/input/output descriptions.
pub fn show_tensor_rgb_captioned(
	img: &Tensor,
	caption: Option<&str>,
	input_desc: Option<&str>,
	output_desc: Option<&str>,
) -> Result<()> {
	// Emit the image as PNG first for the Plot viewer.
	show_tensor_rgb(img)?;
	// Build dims string from the original shape.
	let dims = img.dims();
	let shape = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
	// Grab last-saved image path if any, and emit a single HTML block containing the inline <img> and the caption.
	let saved = last_saved_path().lock().ok().and_then(|g| g.clone());
	let img_html = match tensor_to_png_data_url_rgb(img) {
		Ok(url) => format!(
			"<div><img style=\"max-width:100%; image-rendering: pixelated; border: 1px solid rgba(0,0,0,0.12);\" src=\"{}\"/></div>",
			url
		),
		Err(_) => String::new(),
	};
	let cap_html = caption_html_string(&shape, caption, input_desc, output_desc, saved.as_deref());
	let combined = format!("{}\n{}", img_html, cap_html);
	println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", combined);
	// Plain-text fallback for terminals/viewers that hide HTML captions
	let mut plain = String::new();
	if let Some(c) = caption { if !c.is_empty() { let _ = write!(plain, "Caption: {}  ", c); } }
	let _ = write!(plain, "Dims: {}", shape);
	if let Some(p) = saved.as_deref() { let _ = write!(plain, "  Saved: {}", p.display()); }
	println!("{}", plain);
	Ok(())
}
