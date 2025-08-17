//! Minimal placeholder for an egui-style window inside evcxr.
//! Since running a real native GUI event loop from the Jupyter kernel is fragile,
//! this helper just emits an HTML block framed to look like a window so that
//! notebooks depending on a prior `egui_window` concept can continue to work.
//!
//! If later you want a real interactive GUI, you could gate that behind a feature
//! and use eframe/egui with a native window or WASM target outside the notebook.

use std::time::{SystemTime, UNIX_EPOCH};

/// Open a pseudo-window and render provided HTML body.
/// Arguments:
/// - `title`: Title bar text.
/// - `body_html`: Raw HTML content placed inside the window body region.
///   Returns nothing; prints an HTML block captured by evcxr.
pub fn open(title: &str, body_html: &str) {
    // Provide a unique-ish id so repeated windows don't conflict with element ids.
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let id = format!("eguiwin-{ts}");
    let style = r#"
		box-shadow: 0 4px 16px rgba(0,0,0,0.18);
		border: 1px solid #2d3640;
		background: linear-gradient(#3b4652,#2d3640);
		color: #e9eef2;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, sans-serif;
		border-radius: 6px;
		overflow: hidden;
		margin: 8px 4px 18px 4px;
	"#;
    let title_style = r#"
		background: linear-gradient(#586473,#44505e);
		padding: 4px 10px 6px 10px;
		font-size: 13px;
		letter-spacing: 0.5px;
		font-weight: 500;
		user-select: none;
		border-bottom: 1px solid #1e242b;
	"#;
    let body_style = r#"
		padding: 10px 12px 14px 12px;
		background: #1e242b;
		font-size: 13px;
		line-height: 1.45;
		overflow-x: auto;
		max-height: 420px;
	"#;
    let html = format!(
        r#"
<div id="{id}" style="{style}">
  <div style="{title_style}">{}</div>
  <div style="{body_style}">{}</div>
</div>
"#,
        html_escape::encode_text(title),
        body_html
    );
    println!("EVCXR_BEGIN_CONTENT text/html\n{html}\nEVCXR_END_CONTENT");
}

/// Convenience to open a window with pre-escaped plain text body content.
pub fn open_text(title: &str, body_text: &str) {
    let escaped = html_escape::encode_text(body_text);
    let pre = format!(
        "<pre style=\"margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">{escaped}</pre>"
    );
    open(title, &pre);
}

/// Quick demo content showing a grid of color squares.
pub fn open_color_squares(title: &str) {
    let mut squares = String::new();
    for h in (0..360).step_by(30) {
        // 12 squares
        squares.push_str(&format!(
			"<div style=\"width:32px;height:32px;display:inline-block;margin:3px;border-radius:4px;box-shadow:0 0 0 1px rgba(0,0,0,0.4) inset;background:hsl({h},70%,50%);\"></div>"
		));
    }
    let container = format!("<div>{squares}</div>");
    open(title, &container);
}

/// Try to open a persistent external native window to display content.
/// This is a best-effort helper: when the `external-window` feature is enabled,
/// it will spawn a small `minifb` window showing either a decoded image parsed
/// from a `data:image/png;base64,...` URL in `body_html`, or a simple text raster.
/// If the feature is not enabled or something fails, it falls back to inline HTML via `open`.
pub fn open_external(title: &str, body_html: &str) {
    #[cfg(feature = "external-window")]
    {
        if let Err(e) = crate::egui_window::external::show_minifb_window(title, body_html) {
            eprintln!(
                "[egui_window::open_external] failed to open external window: {e:?}. Falling back to inline HTML."
            );
            open(title, body_html);
        }
        return;
    }
    #[cfg(not(feature = "external-window"))]
    {
        eprintln!(
            "[egui_window::open_external] feature 'external-window' not enabled; rendering inline instead."
        );
        open(title, body_html);
    }
}

#[cfg(feature = "external-window")]
mod external {
    use anyhow::{Context, Result};
    use base64::Engine;
    use image::{DynamicImage, GenericImageView};
    use minifb::{Key, Window, WindowOptions};
    use std::time::Duration;

    pub fn show_minifb_window(title: &str, body_html: &str) -> Result<()> {
        // Very small parser: look for a PNG data-url, decode and display it. Otherwise show text.
        if let Some((w, h, rgba)) = try_extract_png_rgba(body_html).context("parse data-url png")? {
            show_rgba_window(title, w as usize, h as usize, &rgba)
        } else {
            // Render a tiny text message as a placeholder (solid background).
            // For simplicity we just show a solid-color window with a printed log.
            eprintln!(
                "[egui_window::external] No data:image/png found; opening a placeholder window."
            );
            show_placeholder_window(title)
        }
    }

    fn show_rgba_window(title: &str, w: usize, h: usize, rgba: &[u8]) -> Result<()> {
        // Convert RGBA to packed 0xRRGGBB for minifb
        let mut buf = vec![0u32; w * h];
        for (i, px) in buf.iter_mut().enumerate() {
            let r = rgba[i * 4] as u32;
            let g = rgba[i * 4 + 1] as u32;
            let b = rgba[i * 4 + 2] as u32;
            *px = (r << 16) | (g << 8) | b;
        }

        let mut window =
            Window::new(title, w, h, WindowOptions::default()).context("create minifb window")?;
        // target ~60 FPS if supported by the backend
        #[allow(deprecated)]
        {
            window.limit_update_rate(Some(Duration::from_micros(16_666)));
        }

        while window.is_open() && !window.is_key_down(Key::Escape) {
            window
                .update_with_buffer(&buf, w, h)
                .context("update buffer")?;
        }
        Ok(())
    }

    fn show_placeholder_window(title: &str) -> Result<()> {
        let w = 320usize;
        let h = 200usize;
        let mut buf = vec![0u32; w * h];
        // dark slate background
        for px in &mut buf {
            *px = (30u32 << 16) | (36u32 << 8) | 43u32;
        }
        let mut window = Window::new(title, w, h, WindowOptions::default())
            .context("create placeholder window")?;
        #[allow(deprecated)]
        {
            window.limit_update_rate(Some(Duration::from_micros(16_666)));
        }
        while window.is_open() && !window.is_key_down(Key::Escape) {
            window
                .update_with_buffer(&buf, w, h)
                .context("update buffer")?;
        }
        Ok(())
    }

    fn try_extract_png_rgba(body_html: &str) -> Result<Option<(u32, u32, Vec<u8>)>> {
        if let Some(idx) = body_html.find("data:image/png;base64,") {
            let start = idx + "data:image/png;base64,".len();
            // stop at first quote or tag end
            let end = body_html[start..]
                .find(['"', '\'', '>'].as_ref())
                .map(|o| start + o)
                .unwrap_or(body_html.len());
            let b64 = &body_html[start..end];
            let data = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("decode base64 png")?;
            let img = image::load_from_memory(&data).context("decode png")?;
            let (w, h) = img.dimensions();
            let rgba = match img {
                DynamicImage::ImageRgba8(i) => i.into_raw(),
                other => other.to_rgba8().into_raw(),
            };
            return Ok(Some((w, h, rgba)));
        }
        Ok(None)
    }
}
