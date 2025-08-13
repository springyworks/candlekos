use anyhow::{anyhow, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use once_cell::sync::Lazy;
use std::sync::{atomic::{AtomicBool, Ordering}, Mutex};
use std::thread;

use candle_core::{Device, DType, IndexOp, Tensor};

// Public API: show tensors in a single persistent egui window.

// A global flag to allow notebook loops to stop when the user requests.
static STOP_FLAG: AtomicBool = AtomicBool::new(false);

/// Query whether a stop was requested (e.g., via egui window key/button or close).
pub fn should_stop() -> bool {
    STOP_FLAG.load(Ordering::Relaxed)
}

/// Set the stop flag programmatically (e.g., from notebook code).
pub fn set_stop_flag() {
    STOP_FLAG.store(true, Ordering::Relaxed);
}

/// Clear the stop flag so a new loop can run.
pub fn clear_stop_flag() {
    STOP_FLAG.store(false, Ordering::Relaxed);
}

/// Show a grayscale tensor in the persistent egui window.
/// Accepts shapes: [H,W] (2D), [1,H,W], or [H,W,1]. Values are dynamically normalized to 0..255.
pub fn show_tensor_gray(t: &Tensor) -> Result<()> {
    let (rgba, w, h) = tensor_to_rgba_u8_gray(t)?;
    send_image(w, h, rgba)
}

/// Show an RGB tensor in the persistent egui window.
/// Accepts shapes: [3,H,W] (CHW) or [H,W,3] (HWC). Values are dynamically normalized to 0..255.
pub fn show_tensor_rgb(t: &Tensor) -> Result<()> {
    let (rgba, w, h) = tensor_to_rgba_u8_rgb(t)?;
    send_image(w, h, rgba)
}

/// Show a raw RGBA image (u8) in the persistent egui window.
pub fn show_rgba_u8(width: usize, height: usize, rgba: Vec<u8>) -> Result<()> {
    send_image(width, height, rgba)
}

// -------------- Internals: single-window egui app --------------

#[derive(Debug)]
enum Command {
    ShowImage { w: usize, h: usize, rgba: Vec<u8> },
}

static WINDOW_SENDER: Lazy<Mutex<Option<Sender<Command>>>> = Lazy::new(|| Mutex::new(None));

fn send_image(w: usize, h: usize, rgba: Vec<u8>) -> Result<()> {
    ensure_window_spawned();
    let tx = {
        let guard = WINDOW_SENDER
            .lock()
            .map_err(|_| anyhow!("sender mutex poisoned"))?;
        guard
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("egui window not available"))?
    };
    tx.send(Command::ShowImage { w, h, rgba })
        .map_err(|e| anyhow!("failed to send image to egui window: {e}"))
}

fn ensure_window_spawned() {
    let mut guard = WINDOW_SENDER.lock().expect("sender mutex poisoned");
    if guard.is_some() {
        return;
    }

    let (tx, rx) = unbounded::<Command>();
    *guard = Some(tx);

    thread::spawn(move || {
        run_egui_window(rx);
    });
}

fn run_egui_window(rx: Receiver<Command>) {
    #[cfg(target_os = "linux")]
    let native_options = {
        use eframe::NativeOptions;
        // Import both extensions so with_any_thread is available regardless of backend.
        use winit::platform::x11::EventLoopBuilderExtX11;
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        let mut opts = NativeOptions::default();
        opts.event_loop_builder = Some(Box::new(|builder| {
            // Allow creating the event loop off the main thread on Linux.
            winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(builder, true);
            winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(builder, true);
        }));
        opts
    };

    #[cfg(not(target_os = "linux"))]
    let native_options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Egui Plot",
        native_options,
        Box::new(move |_cc| Box::new(EguiPlotApp::new(rx))),
    );
}

struct EguiPlotApp {
    rx: Receiver<Command>,
    texture: Option<egui::TextureHandle>,
    last_size: [usize; 2],
}

impl EguiPlotApp {
    fn new(rx: Receiver<Command>) -> Self {
        Self {
            rx,
            texture: None,
            last_size: [0, 0],
        }
    }
}

impl eframe::App for EguiPlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keyboard shortcuts to request stop.
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Q) || i.key_pressed(egui::Key::Escape) {
                STOP_FLAG.store(true, Ordering::Relaxed);
            }
        });

        // Drain the channel to get the latest image, drop older ones to avoid backlog.
        let mut latest: Option<(usize, usize, Vec<u8>)> = None;
        while let Ok(Command::ShowImage { w, h, rgba }) = self.rx.try_recv() {
            latest = Some((w, h, rgba));
        }

        if let Some((w, h, rgba)) = latest {
            let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            match &mut self.texture {
                Some(tex) => tex.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ctx.load_texture(
                        "plot",
                        image,
                        egui::TextureOptions::LINEAR,
                    ))
                }
            }
            self.last_size = [w, h];
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.label("candle-eguiplot: single-window image viewer");
            if self.last_size != [0, 0] {
                ui.label(format!("{} x {}", self.last_size[0], self.last_size[1]));
            }
            ui.separator();
            ui.horizontal(|ui| {
                let stopped = STOP_FLAG.load(Ordering::Relaxed);
                ui.label(if stopped { "Status: STOP requested" } else { "Status: running" });
                if ui.button("Stop (Q/Esc)").clicked() {
                    STOP_FLAG.store(true, Ordering::Relaxed);
                }
                if ui.button("Clear stop").clicked() {
                    STOP_FLAG.store(false, Ordering::Relaxed);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                if let Some(tex) = &self.texture {
                    let size = tex.size_vec2();
                    // Fit inside panel while keeping aspect ratio.
                    let available = ui.available_size();
                    let scale = (available.x / size.x).min(available.y / size.y).max(1.0);
                    let desired = egui::Vec2::new(size.x * scale, size.y * scale);
                    ui.image((tex.id(), desired));
                } else {
                    ui.label("No image yet. Call show_tensor_* from your notebook.");
                }
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // If the window is closed, request stop so notebook loops can end gracefully.
        STOP_FLAG.store(true, Ordering::Relaxed);
    }
}

// -------------- Tensor -> RGBA conversion helpers --------------

fn tensor_to_rgba_u8_gray(t: &Tensor) -> Result<(Vec<u8>, usize, usize)> {
    let t = if !t.device().is_cpu() {
        t.to_device(&Device::Cpu)?
    } else {
        t.clone()
    };
    let t = if t.dtype() != DType::F32 { t.to_dtype(DType::F32)? } else { t };

    let dims = t.dims();
    let (h, w, mat): (usize, usize, Vec<Vec<f32>>) = match dims {
        [hh, ww] => (*hh, *ww, t.to_vec2::<f32>()?),
        [1, hh, ww] => {
            let squeezed = t.i(0)?; // remove channel dim
            (*hh, *ww, squeezed.to_vec2::<f32>()?)
        }
        [hh, ww, 1] => {
            // Move last dim away by reshaping
            let reshaped = t.reshape(&[*hh, *ww])?;
            (*hh, *ww, reshaped.to_vec2::<f32>()?)
        }
        _ => return Err(anyhow!("gray tensor must be [H,W], [1,H,W], or [H,W,1], got dims={dims:?}")),
    };

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for row in &mat {
        for &v in row {
            if v.is_finite() {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v) <= 0.0 {
        min_v = 0.0;
        max_v = 1.0;
    }
    let scale = 255.0f32 / (max_v - min_v);

    let mut rgba = Vec::with_capacity(w * h * 4);
    for row in mat {
        for v in row {
            let g = (((v - min_v) * scale).clamp(0.0, 255.0)) as u8;
            rgba.extend_from_slice(&[g, g, g, 255]);
        }
    }
    Ok((rgba, w, h))
}

fn tensor_to_rgba_u8_rgb(t: &Tensor) -> Result<(Vec<u8>, usize, usize)> {
    let t = if !t.device().is_cpu() {
        t.to_device(&Device::Cpu)?
    } else {
        t.clone()
    };
    let t = if t.dtype() != DType::F32 { t.to_dtype(DType::F32)? } else { t };

    let dims = t.dims();
    let (is_chw, h, w) = match dims {
        [3, hh, ww] => (true, *hh, *ww),
        [hh, ww, 3] => (false, *hh, *ww),
        _ => return Err(anyhow!("rgb tensor must be [3,H,W] or [H,W,3], got dims={dims:?}")),
    };

    // Extract 3 channels as 2D f32 mats
    let (r2, g2, b2) = if is_chw {
        let r = t.i(0)?;
        let g = t.i(1)?;
        let b = t.i(2)?;
        (r.to_vec2::<f32>()?, g.to_vec2::<f32>()?, b.to_vec2::<f32>()?)
    } else {
        // HWC: need to iterate and collect
        let v3 = t.to_vec3::<f32>()?; // [H][W][3]
        let mut r2 = vec![vec![0f32; w]; h];
        let mut g2 = vec![vec![0f32; w]; h];
        let mut b2 = vec![vec![0f32; w]; h];
        for y in 0..h {
            for x in 0..w {
                r2[y][x] = v3[y][x][0];
                g2[y][x] = v3[y][x][1];
                b2[y][x] = v3[y][x][2];
            }
        }
        (r2, g2, b2)
    };

    // Dynamic range per-channel
    let (rmin, rmax) = min_max_2d(&r2);
    let (gmin, gmax) = min_max_2d(&g2);
    let (bmin, bmax) = min_max_2d(&b2);
    let rscale = if rmax > rmin { 255.0 / (rmax - rmin) } else { 1.0 };
    let gscale = if gmax > gmin { 255.0 / (gmax - gmin) } else { 1.0 };
    let bscale = if bmax > bmin { 255.0 / (bmax - bmin) } else { 1.0 };

    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let r = (((r2[y][x] - rmin) * rscale).clamp(0.0, 255.0)) as u8;
            let g = (((g2[y][x] - gmin) * gscale).clamp(0.0, 255.0)) as u8;
            let b = (((b2[y][x] - bmin) * bscale).clamp(0.0, 255.0)) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }

    Ok((rgba, w, h))
}

fn min_max_2d(m: &Vec<Vec<f32>>) -> (f32, f32) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for row in m.iter() {
        for &v in row.iter() {
            if v.is_finite() {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        (0.0, 1.0)
    } else {
        (min_v, max_v)
    }
}
