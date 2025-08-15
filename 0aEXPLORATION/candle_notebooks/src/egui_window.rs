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
/// Returns nothing; prints an HTML block captured by evcxr.
pub fn open(title: &str, body_html: &str) {
	// Provide a unique-ish id so repeated windows don't conflict with element ids.
	let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
	let id = format!("eguiwin-{}", ts);
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
	let html = format!(r#"
<div id="{id}" style="{style}">
  <div style="{title_style}">{}</div>
  <div style="{body_style}">{}</div>
</div>
"#, html_escape::encode_text(title), body_html);
	println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", html);
}

/// Convenience to open a window with pre-escaped plain text body content.
pub fn open_text(title: &str, body_text: &str) {
	let escaped = html_escape::encode_text(body_text);
	let pre = format!("<pre style=\"margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">{}</pre>", escaped);
	open(title, &pre);
}

/// Quick demo content showing a grid of color squares.
pub fn open_color_squares(title: &str) {
	let mut squares = String::new();
	for h in (0..360).step_by(30) { // 12 squares
		squares.push_str(&format!(
			"<div style=\"width:32px;height:32px;display:inline-block;margin:3px;border-radius:4px;box-shadow:0 0 0 1px rgba(0,0,0,0.4) inset;background:hsl({h},70%,50%);\"></div>"
		));
	}
	let container = format!("<div>{}</div>", squares);
	open(title, &container);
}

