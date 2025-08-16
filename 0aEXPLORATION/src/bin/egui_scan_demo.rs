//! egui_scan_demo: Display two tensor panes. Left: evolving synthetic tensor. Right: inclusive scan over a selected axis.
//! Runs on CPU by default; enable `--features cuda` to use GPU if available.

use candle_core::{DType, Device, Result, Tensor};
use candle_exploration::proc_fields::{gray, plasma, sinusoidal_mix, tensor_to_rgba};
use eframe::egui;
use std::time::{Duration, Instant};

const W: usize = 128;
const H: usize = 128;

struct ScanApp {
    device: Device,
    start: Instant,
    last_update: Instant,
    frame: usize,
    base_tensor: Tensor, // [H, W]
    scan_tensor: Tensor, // [H, W]
    axis: usize,
    speed: f32,
    paused: bool,
    use_exclusive: bool,
    colormap: Colormap,
}

#[derive(Copy, Clone, PartialEq)]
enum Colormap {
    Gray,
    Plasma,
}

impl ScanApp {
    fn new(device: Device) -> Result<Self> {
        let zero = Tensor::zeros((H, W), DType::F32, &device)?;
        Ok(Self {
            device,
            start: Instant::now(),
            last_update: Instant::now(),
            frame: 0,
            base_tensor: zero.clone(),
            scan_tensor: zero,
            axis: 1,
            speed: 1.0,
            paused: false,
            use_exclusive: false,
            colormap: Colormap::Gray,
        })
    }

    fn synthetic(&self, t: f32) -> Result<Tensor> {
        // Delegates to procedural field helper (radial + linear sinusoid blend)
        sinusoidal_mix(H, W, t, 10.0, 5.0, 3.0, &self.device)
    }

    fn update_tensors(&mut self) {
        if self.paused {
            return;
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_update);
        // limit update rate (~60 Hz)
        if dt < Duration::from_millis(16) {
            return;
        }
        self.last_update = now;
        let elapsed = now.duration_since(self.start).as_secs_f32() * self.speed;
        self.frame += 1;
        if let Ok(base) = self.synthetic(elapsed) {
            self.base_tensor = base;
            self.scan_tensor = if self.use_exclusive {
                // add a nominal batch dim to reuse existing API expectations if needed
                match self.base_tensor.exclusive_scan(self.axis) {
                    Ok(t) => t,
                    Err(_) => self.base_tensor.clone(),
                }
            } else {
                match self.base_tensor.inclusive_scan(self.axis) {
                    Ok(t) => t,
                    Err(_) => self.base_tensor.clone(),
                }
            };
        }
    }

    fn to_image_pixels(&self, t: &Tensor) -> Vec<u8> {
        match self.colormap {
            Colormap::Gray => tensor_to_rgba(t, gray),
            Colormap::Plasma => tensor_to_rgba(t, plasma),
        }
    }
}

impl eframe::App for ScanApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_tensors();
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.label(format!("frame: {}", self.frame));
            ui.horizontal(|ui| {
                ui.label("axis");
                if ui.add(egui::Slider::new(&mut self.axis, 0..=1)).changed() {
                    // axis changed -> recompute scan immediately
                    self.last_update = Instant::now() - Duration::from_millis(17);
                }
                ui.checkbox(&mut self.use_exclusive, "exclusive");
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=4.0).text("speed"));
                ui.checkbox(&mut self.paused, "paused");
                ui.radio_value(&mut self.colormap, Colormap::Gray, "gray");
                ui.radio_value(&mut self.colormap, Colormap::Plasma, "plasma");
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let left = self.to_image_pixels(&self.base_tensor);
                let right = self.to_image_pixels(&self.scan_tensor);
                let img_left = egui::ColorImage::from_rgba_unmultiplied([W, H], &left);
                let img_right = egui::ColorImage::from_rgba_unmultiplied([W, H], &right);
                let tex_left =
                    ui.ctx()
                        .load_texture("left", img_left, egui::TextureOptions::NEAREST);
                let tex_right =
                    ui.ctx()
                        .load_texture("right", img_right, egui::TextureOptions::NEAREST);
                ui.vertical(|ui| {
                    ui.label("source");
                    ui.image(&tex_left);
                });
                ui.vertical(|ui| {
                    ui.label(format!("scan axis {}", self.axis));
                    ui.image(&tex_right);
                });
            });
        });
        ctx.request_repaint();
    }
}

// (colormaps provided by proc_fields)

fn main() -> Result<()> {
    let device = if std::env::var("CANDLE_FORCE_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "egui_scan_demo",
        native_options,
        Box::new(|_cc| Box::new(ScanApp::new(device).expect("init app"))),
    )
    .unwrap();
    Ok(())
}
