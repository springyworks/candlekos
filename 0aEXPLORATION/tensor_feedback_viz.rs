use candle_core::{Device, Result, Tensor};
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};

const WINDOW_WIDTH: usize = 1200;
const WINDOW_HEIGHT: usize = 600;
const TENSOR_WIDTH: usize = 256;
const TENSOR_HEIGHT: usize = 256;
const DISPLAY_WIDTH: usize = WINDOW_WIDTH / 2;

// Feature-gated debug macro: enable with --features viz-debug
#[cfg(feature = "viz-debug")]
macro_rules! viz_debug { ($($t:tt)*) => { eprintln!("[viz-debug] {}", format!($($t)*)); } }
#[cfg(not(feature = "viz-debug"))]
macro_rules! viz_debug { ($($t:tt)*) => {}; }

struct TensorClosedLoopViz {
    window: Window,
    device: Device,
    
    // Two tensors for closed loop system
    tensor_a: Tensor,  // "Sensor A"
    tensor_b: Tensor,  // "Sensor B" 
    
    // Parameters
    time: f32,
    coupling_strength: f32,
    rotation_speed: f32,
    zoom_factor: f32,
    noise_level: f32,
    
    // Processing modes
    mode: ProcessingMode,
    filter_type: FilterType,
    
    // Animation state
    phase: f32,
    decay_rate: f32,
    divergence_strength: f32,
    // For reporting changes without spamming
    last_report_coupling: f32,
    last_report_rotation: f32,
    last_report_mode: Option<ProcessingMode>,
    last_report_filter: Option<FilterType>,
    last_report_noise: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcessingMode {
    DirectCoupling,    // B = transform(A)
    CrossCoupling,     // A ↔ B with transforms
    Interference,      // A = A + transform(B), B = B + transform(A)
    Convolution,       // Use conv2d for coupling
    FFTCoupling,       // Use 2D FFT for coupling
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterType {
    None,
    Gaussian,
    Sobel,
    Laplacian,
    Emboss,
}

impl TensorClosedLoopViz {
    // Debug helper: safely obtain a flat Vec<f32> from any tensor while printing shape info.
    fn debug_flat_vec(&self, tensor: &Tensor, label: &str) -> Result<Vec<f32>> {
        let shape = tensor.shape();
        let dims = shape.dims();
        if dims.len() != 1 {
            viz_debug!("{} shape {:?} (rank {}) -> flatten", label, dims, dims.len());
            let flat = tensor.flatten_all()?;
            viz_debug!("{} flattened shape {:?}", label, flat.shape().dims());
            return flat.to_vec1::<f32>();
        }
        viz_debug!("{} already flat shape {:?}", label, dims);
        tensor.to_vec1::<f32>()
    }
    fn new() -> Result<Self> {
        let window = Window::new(
            "🔄 Candle Tensor Closed Loop System - [1-5] Modes [F1-F5] Filters [WASD] Controls [R] Reset",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WindowOptions::default(),
        ).unwrap();

        let device = Device::Cpu;
        
        // Initialize tensors with interesting patterns
        let tensor_a = Self::create_initial_pattern(&device, 0)?;
        let tensor_b = Self::create_initial_pattern(&device, 1)?;

        Ok(Self {
            window,
            device,
            tensor_a,
            tensor_b,
            time: 0.0,
            coupling_strength: 0.3,
            rotation_speed: 0.1,
            zoom_factor: 1.02,
            noise_level: 0.01,
            mode: ProcessingMode::DirectCoupling,
            filter_type: FilterType::None,
            phase: 0.0,
            decay_rate: 0.995,
            divergence_strength: 0.15,
            last_report_coupling: 0.3,
            last_report_rotation: 0.1,
            last_report_mode: None,
            last_report_filter: None,
            last_report_noise: true, // noise_level > 0
        })
    }
    
    fn create_initial_pattern(device: &Device, pattern_type: i32) -> Result<Tensor> {
        let mut data = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        let center_x = TENSOR_WIDTH as f32 / 2.0;
        let center_y = TENSOR_HEIGHT as f32 / 2.0;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let r = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);
                
                let value = match pattern_type {
                    0 => {
                        // Spiral pattern
                        let spiral = (r * 0.1 + angle * 2.0).sin() * 0.5 + 0.5;
                        spiral * (-r * 0.01).exp()
                    },
                    1 => {
                        // Concentric circles with interference
                        let circles = (r * 0.2).sin() * (r * 0.15 + angle).cos();
                        (circles * 0.5 + 0.5) * (-r * 0.008).exp()
                    },
                    _ => {
                        // Default: simple gradient
                        (x as f32 / TENSOR_WIDTH as f32) * (y as f32 / TENSOR_HEIGHT as f32)
                    }
                };
                
                data[y * TENSOR_WIDTH + x] = value;
            }
        }
        
        Tensor::from_vec(data, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], device)
    }
    
    fn apply_transform(&self, input: &Tensor) -> Result<Tensor> {
        match self.mode {
            ProcessingMode::DirectCoupling => {
                self.apply_geometric_transform(input)
            },
            ProcessingMode::CrossCoupling => {
                let rotated = self.apply_rotation(input)?;
                self.apply_zoom(&rotated)
            },
            ProcessingMode::Interference => {
                let fft_transformed = self.apply_fft_transform(input)?;
                self.apply_frequency_filter(&fft_transformed)
            },
            ProcessingMode::Convolution => {
                self.apply_convolution(input)
            },
            ProcessingMode::FFTCoupling => {
                self.apply_fft_transform(input)
            },
        }
    }
    
    fn apply_geometric_transform(&self, input: &Tensor) -> Result<Tensor> {
        let squeezed = input.squeeze(0)?.squeeze(0)?;
        let data = self.debug_flat_vec(&squeezed, "apply_geometric_transform.input")?;
        let mut output = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        let center_x = TENSOR_WIDTH as f32 / 2.0;
        let center_y = TENSOR_HEIGHT as f32 / 2.0;
        let angle = self.time * self.rotation_speed;
        let zoom = self.zoom_factor;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                // Apply rotation and zoom around center
                let dx = (x as f32 - center_x) / zoom;
                let dy = (y as f32 - center_y) / zoom;
                
                let rotated_x = dx * angle.cos() - dy * angle.sin() + center_x;
                let rotated_y = dx * angle.sin() + dy * angle.cos() + center_y;
                
                // Bilinear interpolation
                if rotated_x >= 0.0 && rotated_x < TENSOR_WIDTH as f32 - 1.0 &&
                   rotated_y >= 0.0 && rotated_y < TENSOR_HEIGHT as f32 - 1.0 {
                    
                    let x0 = rotated_x.floor() as usize;
                    let y0 = rotated_y.floor() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    
                    let fx = rotated_x - x0 as f32;
                    let fy = rotated_y - y0 as f32;
                    
                    let v00 = data[y0 * TENSOR_WIDTH + x0];
                    let v10 = data[y0 * TENSOR_WIDTH + x1];
                    let v01 = data[y1 * TENSOR_WIDTH + x0];
                    let v11 = data[y1 * TENSOR_WIDTH + x1];
                    
                    let interpolated = v00 * (1.0 - fx) * (1.0 - fy) +
                                     v10 * fx * (1.0 - fy) +
                                     v01 * (1.0 - fx) * fy +
                                     v11 * fx * fy;
                    
                    output[y * TENSOR_WIDTH + x] = interpolated * self.decay_rate;
                }
            }
        }
        
        // Add some noise for interesting dynamics
        if self.noise_level > 0.0 {
            for i in 0..output.len() {
                let noise = (fastrand::f32() - 0.5) * self.noise_level;
                output[i] = (output[i] + noise).clamp(0.0, 1.0);
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn apply_rotation(&self, input: &Tensor) -> Result<Tensor> {
        // Simple 90-degree rotation for cross-coupling
        let squeezed = input.squeeze(0)?.squeeze(0)?;
        let data = self.debug_flat_vec(&squeezed, "apply_rotation.input")?;
        let mut output = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let new_x = TENSOR_HEIGHT - 1 - y;
                let new_y = x;
                if new_x < TENSOR_WIDTH && new_y < TENSOR_HEIGHT {
                    output[new_y * TENSOR_WIDTH + new_x] = data[y * TENSOR_WIDTH + x];
                }
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn apply_zoom(&self, input: &Tensor) -> Result<Tensor> {
        let squeezed = input.squeeze(0)?.squeeze(0)?;
        let data = self.debug_flat_vec(&squeezed, "apply_zoom.input")?;
        let mut output = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        let zoom = 1.0 + 0.1 * (self.time * 0.5).sin();
        let center_x = TENSOR_WIDTH as f32 / 2.0;
        let center_y = TENSOR_HEIGHT as f32 / 2.0;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let src_x = ((x as f32 - center_x) / zoom + center_x) as i32;
                let src_y = ((y as f32 - center_y) / zoom + center_y) as i32;
                
                if src_x >= 0 && src_x < TENSOR_WIDTH as i32 && 
                   src_y >= 0 && src_y < TENSOR_HEIGHT as i32 {
                    output[y * TENSOR_WIDTH + x] = data[src_y as usize * TENSOR_WIDTH + src_x as usize];
                }
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn apply_fft_transform(&self, input: &Tensor) -> Result<Tensor> {
        // Apply 2D FFT, modify frequency domain, then inverse FFT
        let squeezed = input.squeeze(0)?.squeeze(0)?; // Remove batch and channel dims to get [H, W]
        
        // Apply 2D FFT if available, otherwise use magnitude directly
        match squeezed.fft2(true, false) {
            Ok(fft_result) => {
                let magnitude = fft_result.fft_magnitude()?;
                // Modify frequency domain - apply a frequency filter
                let modified = self.apply_frequency_filter(&magnitude)?;
                Ok(modified.unsqueeze(0)?.unsqueeze(0)?)
            }
            Err(_) => {
                // Fallback to simple processing if FFT2 fails
                let data = squeezed.flatten_all()?.to_vec1::<f32>()?;
                let processed: Vec<f32> = data.iter()
                    .enumerate()
                    .map(|(i, &x)| x * (1.0 + 0.1 * (self.time + i as f32 * 0.01).sin()))
                    .collect();
                let result = Tensor::from_vec(processed, squeezed.shape(), &self.device)?;
                Ok(result.unsqueeze(0)?.unsqueeze(0)?)
            }
        }
    }
    
    fn apply_frequency_filter(&self, freq_tensor: &Tensor) -> Result<Tensor> {
    let data = freq_tensor.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0f32; data.len()];
        
        let center_x = TENSOR_WIDTH / 2;
        let center_y = TENSOR_HEIGHT / 2;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let dx = x as i32 - center_x as i32;
                let dy = y as i32 - center_y as i32;
                let freq = ((dx * dx + dy * dy) as f32).sqrt();
                
                // Apply frequency-dependent filter
                let filter_value = match self.filter_type {
                    FilterType::None => 1.0,
                    FilterType::Gaussian => (-freq * freq / 1000.0).exp(),
                    FilterType::Sobel => if freq > 10.0 && freq < 50.0 { 1.0 } else { 0.1 },
                    FilterType::Laplacian => freq * freq / 10000.0,
                    FilterType::Emboss => if freq > 5.0 { (freq / 20.0).sin() } else { 0.0 },
                };
                
                output[y * TENSOR_WIDTH + x] = data[y * TENSOR_WIDTH + x] * filter_value;
            }
        }
        
        Tensor::from_vec(output, &[TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn apply_convolution(&self, input: &Tensor) -> Result<Tensor> {
        // Apply different convolution kernels
        let kernel = match self.filter_type {
            FilterType::Gaussian => vec![
                0.0625, 0.125, 0.0625,
                0.125,  0.25,  0.125,
                0.0625, 0.125, 0.0625,
            ],
            FilterType::Sobel => vec![
                -1.0, 0.0, 1.0,
                -2.0, 0.0, 2.0,
                -1.0, 0.0, 1.0,
            ],
            FilterType::Laplacian => vec![
                0.0, -1.0, 0.0,
                -1.0, 4.0, -1.0,
                0.0, -1.0, 0.0,
            ],
            FilterType::Emboss => vec![
                -2.0, -1.0, 0.0,
                -1.0,  1.0, 1.0,
                0.0,   1.0, 2.0,
            ],
            FilterType::None => vec![
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        };
        
        self.apply_kernel_convolution(input, &kernel)
    }
    
    fn apply_kernel_convolution(&self, input: &Tensor, kernel: &[f32]) -> Result<Tensor> {
    let data = input.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        for y in 1..TENSOR_HEIGHT-1 {
            for x in 1..TENSOR_WIDTH-1 {
                let mut sum = 0.0;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let py = y + ky - 1;
                        let px = x + kx - 1;
                        sum += data[py * TENSOR_WIDTH + px] * kernel[ky * 3 + kx];
                    }
                }
                output[y * TENSOR_WIDTH + x] = sum.clamp(0.0, 1.0);
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn apply_fft_coupling_with_params(&self, input: &Tensor, energy: f32, phase: f32) -> Result<Tensor> {
        // Complex FFT-based coupling with phase manipulation
        let squeezed = input.squeeze(0)?.squeeze(0)?; // Get [H, W]
        
        // Apply 1D FFT along width dimension for each row
        match squeezed.rfft(1, false) {
            Ok(fft_result) => {
                let magnitude = fft_result.fft_magnitude()?;
                let phase_result = fft_result.fft_phase()?;
                
                // Modify phase based on time and coupling
                let phase_data = phase_result.flatten_all()?.to_vec1::<f32>()?;
                let modified_phase: Vec<f32> = phase_data.iter()
                    .enumerate()
                    .map(|(i, &p)| p + phase * 0.1 + (i as f32 * 0.01).sin() * energy)
                    .collect();
                
                let _new_phase = Tensor::from_vec(modified_phase, phase_result.shape(), &self.device)?;
                
                // Reconstruct and return (simplified - just use magnitude for now)
                Ok(magnitude.unsqueeze(0)?.unsqueeze(0)?)
            }
            Err(_) => {
                // Fallback to simple time-based modulation
                let data = squeezed.flatten_all()?.to_vec1::<f32>()?;
                let processed: Vec<f32> = data.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let local_phase = phase * 2.0 + i as f32 * 0.02;
                        x * (0.8 + 0.2 * local_phase.sin() * energy)
                    })
                    .collect();
                let result = Tensor::from_vec(processed, squeezed.shape(), &self.device)?;
                Ok(result.unsqueeze(0)?.unsqueeze(0)?)
            }
        }
    }
    
    // Create modulation pattern from another tensor's characteristics
    fn create_modulation_from_tensor(&self, tensor: &Tensor, phase: f32) -> Result<Vec<f32>> {
    let squeezed = tensor.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "create_modulation_from_tensor.tensor")?;
        let mut modulation = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let idx = y * TENSOR_WIDTH + x;
                let value = data[idx];
                
                // Create spatial modulation pattern based on tensor value
                let dx = x as f32 / TENSOR_WIDTH as f32 - 0.5;
                let dy = y as f32 / TENSOR_HEIGHT as f32 - 0.5;
                let r = (dx * dx + dy * dy).sqrt();
                
                let mod_value = (value * 5.0 + r * 10.0 + phase).sin() * 0.5 + 0.5;
                modulation[idx] = mod_value;
            }
        }
        
        Ok(modulation)
    }
    
    // Apply tensor modulation (element-wise multiplication with modulation pattern)
    fn apply_tensor_modulation(&self, tensor: &Tensor, modulation: &[f32], strength: f32) -> Result<Tensor> {
    let squeezed = tensor.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "apply_tensor_modulation.tensor")?;
        let mut output = vec![0.0f32; data.len()];
        
        for i in 0..data.len() {
            let mod_effect = 1.0 - strength + strength * modulation[i];
            output[i] = data[i] * mod_effect;
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    // Apply transformation with dynamic parameters
    fn apply_transform_with_params(&self, input: &Tensor, rotation: f32, zoom: f32) -> Result<Tensor> {
    let squeezed = input.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "apply_transform_with_params.input")?;
        let mut output = vec![0.0f32; data.len()];
        
        let center_x = TENSOR_WIDTH as f32 / 2.0;
        let center_y = TENSOR_HEIGHT as f32 / 2.0;
        let angle = self.time * rotation;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                // Apply rotation and zoom around center with dynamic parameters
                let dx = (x as f32 - center_x) / zoom;
                let dy = (y as f32 - center_y) / zoom;
                
                let rotated_x = dx * angle.cos() - dy * angle.sin() + center_x;
                let rotated_y = dx * angle.sin() + dy * angle.cos() + center_y;
                
                // Bilinear interpolation
                if rotated_x >= 0.0 && rotated_x < TENSOR_WIDTH as f32 - 1.0 &&
                   rotated_y >= 0.0 && rotated_y < TENSOR_HEIGHT as f32 - 1.0 {
                    
                    let x0 = rotated_x.floor() as usize;
                    let y0 = rotated_y.floor() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    
                    let fx = rotated_x - x0 as f32;
                    let fy = rotated_y - y0 as f32;
                    
                    let v00 = data[y0 * TENSOR_WIDTH + x0];
                    let v10 = data[y0 * TENSOR_WIDTH + x1];
                    let v01 = data[y1 * TENSOR_WIDTH + x0];
                    let v11 = data[y1 * TENSOR_WIDTH + x1];
                    
                    let interpolated = v00 * (1.0 - fx) * (1.0 - fy) +
                                     v10 * fx * (1.0 - fy) +
                                     v01 * (1.0 - fx) * fy +
                                     v11 * fx * fy;
                    
                    output[y * TENSOR_WIDTH + x] = interpolated * self.decay_rate;
                }
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    // Extract features from a tensor to influence another
    fn extract_tensor_features(&self, tensor: &Tensor) -> Result<Vec<f32>> {
    let squeezed = tensor.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "extract_tensor_features.tensor")?;
        let mut features = vec![0.0f32; 9]; // 3x3 feature map
        
        // Extract average values from 9 regions of the tensor
        let region_w = TENSOR_WIDTH / 3;
        let region_h = TENSOR_HEIGHT / 3;
        
        for j in 0..3 {
            for i in 0..3 {
                let mut sum = 0.0;
                let mut count = 0;
                
                for y in j * region_h..(j + 1) * region_h {
                    for x in i * region_w..(i + 1) * region_w {
                        sum += data[y * TENSOR_WIDTH + x];
                        count += 1;
                    }
                }
                
                features[j * 3 + i] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
        
        Ok(features)
    }
    
    // Modulate tensor using features from another tensor
    fn modulate_tensor_by_features(&self, tensor: &Tensor, features: &[f32], phase: f32) -> Result<Tensor> {
    let squeezed = tensor.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "modulate_tensor_by_features.tensor")?;
        let mut output = vec![0.0f32; data.len()];
        
        let region_w = TENSOR_WIDTH / 3;
        let region_h = TENSOR_HEIGHT / 3;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let region_i = (x / region_w).min(2);
                let region_j = (y / region_h).min(2);
                let feature = features[region_j * 3 + region_i];
                
                // Apply feature-based modulation with phase
                let mod_value = 0.7 + 0.3 * (phase + feature * 5.0).sin();
                output[y * TENSOR_WIDTH + x] = data[y * TENSOR_WIDTH + x] * mod_value;
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    // Create wave pattern from tensor for interference mode
    fn create_wave_pattern_from_tensor(&self, tensor: &Tensor, phase: f32) -> Result<Tensor> {
    let squeezed = tensor.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "create_wave_pattern_from_tensor.tensor")?;
        let mut output = vec![0.0f32; data.len()];
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let value = data[y * TENSOR_WIDTH + x];
                let dx = x as f32 - TENSOR_WIDTH as f32 / 2.0;
                let dy = y as f32 - TENSOR_HEIGHT as f32 / 2.0;
                let distance = (dx * dx + dy * dy).sqrt();
                
                // Create wave pattern modulated by tensor value
                let wave = (distance * 0.1 * value + phase).sin();
                output[y * TENSOR_WIDTH + x] = wave * 0.5 + 0.5;
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    // Create interference pattern between two wave tensors
    fn create_interference_pattern(&self, wave_a: &Tensor, wave_b: &Tensor) -> Result<Tensor> {
    let data_a = wave_a.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
    let data_b = wave_b.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0f32; data_a.len()];
        
        for i in 0..data_a.len() {
            // Convert values from [0,1] to [-1,1] for proper wave interference
            let a = data_a[i] * 2.0 - 1.0;
            let b = data_b[i] * 2.0 - 1.0;
            
            // Create interference (constructive + destructive)
            let interference = a + b + a * b * 0.5;
            
            // Back to [0,1] range
            output[i] = (interference * 0.25 + 0.5).clamp(0.0, 1.0);
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    // Extract convolution kernel from tensor regions
    fn extract_kernel_from_tensor(&self, tensor: &Tensor, filter_type: FilterType) -> Result<Vec<f32>> {
    let data = tensor.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let mut kernel = vec![0.0f32; 9];
        
        // Extract 3x3 kernel from tensor center or dominant region
        let center_x = TENSOR_WIDTH / 2;
        let center_y = TENSOR_HEIGHT / 2;
        let kernel_size = 3;
        let offset = kernel_size / 2;
        
        let mut sum = 0.0;
        for ky in 0..kernel_size {
            for kx in 0..kernel_size {
                let x = center_x + kx - offset;
                let y = center_y + ky - offset;
                if x < TENSOR_WIDTH && y < TENSOR_HEIGHT {
                    let value = data[y * TENSOR_WIDTH + x];
                    kernel[ky * kernel_size + kx] = value;
                    sum += value;
                }
            }
        }
        
        // Normalize kernel based on filter type
        match filter_type {
            FilterType::Gaussian => {
                // Ensure sum is 1 for blur kernel
                if sum > 0.0 {
                    for i in 0..kernel.len() {
                        kernel[i] /= sum;
                    }
                }
            },
            FilterType::Sobel => {
                // Make edge detection kernel
                let center_value = kernel[4];
                for i in 0..kernel.len() {
                    if i == 4 { // center
                        kernel[i] = 8.0 * center_value;
                    } else {
                        kernel[i] = -center_value;
                    }
                }
            },
            FilterType::Laplacian => {
                // Laplacian kernel
                let center_value = kernel[4];
                for i in 0..kernel.len() {
                    if i == 4 { // center
                        kernel[i] = 4.0 * center_value;
                    } else if i % 2 == 0 { // corners
                        kernel[i] = -0.5 * center_value;
                    } else { // edges
                        kernel[i] = -center_value;
                    }
                }
            },
            FilterType::Emboss => {
                // Emboss kernel
                for i in 0..kernel.len() {
                    if i < 4 {
                        kernel[i] = -kernel[i];
                    } else if i > 4 {
                        kernel[i] = kernel[i];
                    } else {
                        kernel[i] = 1.0; // Center value
                    }
                }
            },
            FilterType::None => {
                // Identity kernel
                for i in 0..kernel.len() {
                    kernel[i] = if i == 4 { 1.0 } else { 0.0 };
                }
            },
        }
        
        Ok(kernel)
    }
    
    // Extract frequency modulation pattern from FFT result
    fn extract_frequency_modulation(&self, fft_tensor: &Tensor, energy: f32) -> Result<Vec<f32>> {
    // We only need shape; no direct use of underlying values here so skip flatten to avoid cost.
        let mut modulation = vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT];
        
        // Create frequency modulation pattern
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let dx = x as f32 / TENSOR_WIDTH as f32 - 0.5;
                let dy = y as f32 / TENSOR_HEIGHT as f32 - 0.5;
                let r = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);
                
                // Create modulation based on polar coordinates and energy
                let mod_value = (r * 10.0 * energy + angle + self.time).sin() * 0.5 + 0.5;
                modulation[y * TENSOR_WIDTH + x] = mod_value;
            }
        }
        
        Ok(modulation)
    }
    
    // Apply frequency domain modulation to tensor
    fn apply_frequency_modulation(&self, tensor: &Tensor, modulation: &[f32], strength: f32) -> Result<Tensor> {
    let data = tensor.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0f32; data.len()];
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let idx = y * TENSOR_WIDTH + x;
                let mod_value = modulation[idx];
                
                // Apply modulation with dynamic strength
                let effect = 1.0 - strength + strength * mod_value;
                output[idx] = data[idx] * effect;
            }
        }
        
        Tensor::from_vec(output, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn update_feedback(&mut self) -> Result<()> {
        // Energy-based dynamics create a closed-loop parameter modulation
    let tensor_a_energy = self.tensor_a.flatten_all()?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let tensor_b_energy = self.tensor_b.flatten_all()?.sqr()?.sum_all()?.to_scalar::<f32>()?;

        let norm_a_energy = (tensor_a_energy / (TENSOR_HEIGHT * TENSOR_WIDTH) as f32).clamp(0.0, 1.0);
        let norm_b_energy = (tensor_b_energy / (TENSOR_HEIGHT * TENSOR_WIDTH) as f32).clamp(0.0, 1.0);

        let dynamic_rotation = self.rotation_speed * (1.0 + norm_b_energy);
        let dynamic_zoom = self.zoom_factor * (1.0 + norm_a_energy * 0.2);
        let dynamic_coupling = self.coupling_strength * (0.5 + norm_a_energy * norm_b_energy);

        let phase_shift_a = self.phase + norm_b_energy * 2.0 + self.time * 0.1;
        let phase_shift_b = self.phase - norm_a_energy * 2.0 + self.time * 0.1;

        // Save originals for cross operations
        let orig_a = self.tensor_a.clone();
        let orig_b = self.tensor_b.clone();

        match self.mode {
            ProcessingMode::DirectCoupling => {
                let transformed_a = self.apply_transform_with_params(&orig_a, dynamic_rotation, dynamic_zoom)?;
                let transformed_b = self.apply_transform_with_params(&orig_b, -dynamic_rotation * 0.8, dynamic_zoom)?;
                self.tensor_a = self.blend_tensors(&transformed_a, &orig_b, dynamic_coupling * 0.6)?;
                self.tensor_b = self.blend_tensors(&transformed_b, &orig_a, dynamic_coupling * 0.6)?;
            }
            ProcessingMode::CrossCoupling => {
                let feats_a = self.extract_tensor_features(&orig_a)?;
                let feats_b = self.extract_tensor_features(&orig_b)?;
                let mod_a = self.modulate_tensor_by_features(&orig_a, &feats_b, phase_shift_a)?;
                let mod_b = self.modulate_tensor_by_features(&orig_b, &feats_a, phase_shift_b)?;
                self.tensor_a = self.blend_tensors(&orig_a, &mod_b, dynamic_coupling)?;
                self.tensor_b = self.blend_tensors(&orig_b, &mod_a, dynamic_coupling)?;
            }
            ProcessingMode::Interference => {
                let wave_a = self.create_wave_pattern_from_tensor(&orig_a, phase_shift_a)?;
                let wave_b = self.create_wave_pattern_from_tensor(&orig_b, phase_shift_b)?;
                let interference = self.create_interference_pattern(&wave_a, &wave_b)?;
                self.tensor_a = self.blend_tensors(&orig_a, &interference, dynamic_coupling * 0.6)?;
                self.tensor_b = self.blend_tensors(&orig_b, &interference, dynamic_coupling * 0.5)?;
            }
            ProcessingMode::Convolution => {
                let kernel_a = self.extract_kernel_from_tensor(&orig_b, self.filter_type)?;
                let kernel_b = self.extract_kernel_from_tensor(&orig_a, self.filter_type)?;
                let conv_a = self.apply_kernel_convolution(&orig_a, &kernel_b)?;
                let conv_b = self.apply_kernel_convolution(&orig_b, &kernel_a)?;
                self.tensor_a = self.blend_tensors(&orig_a, &conv_b, dynamic_coupling * 0.8)?;
                self.tensor_b = self.blend_tensors(&orig_b, &conv_a, dynamic_coupling)?;
            }
            ProcessingMode::FFTCoupling => {
                let fft_a = self.apply_fft_coupling_with_params(&orig_a, norm_b_energy, phase_shift_a)?;
                let fft_b = self.apply_fft_coupling_with_params(&orig_b, norm_a_energy, phase_shift_b)?;
                let freq_mod_a = self.extract_frequency_modulation(&fft_b, norm_a_energy)?;
                let freq_mod_b = self.extract_frequency_modulation(&fft_a, norm_b_energy)?;
                let mod_a = self.apply_frequency_modulation(&orig_a, &freq_mod_b, dynamic_coupling)?;
                let mod_b = self.apply_frequency_modulation(&orig_b, &freq_mod_a, dynamic_coupling * 0.9)?;
                self.tensor_a = mod_a;
                self.tensor_b = mod_b;
            }
        }

        // Gentle decay + noise to keep patterns evolving
        let decay = self.decay_rate;
        let noise_scale = (self.noise_level * 0.1).max(1e-4);
        let noise_a = Tensor::randn(0.0, noise_scale, &[1,1,TENSOR_HEIGHT,TENSOR_WIDTH], &self.device)?;
        let noise_b = Tensor::randn(0.0, noise_scale, &[1,1,TENSOR_HEIGHT,TENSOR_WIDTH], &self.device)?;
    let decay_tensor = Tensor::full(decay, self.tensor_a.shape(), &self.device)?;
    self.tensor_a = ((&self.tensor_a * &decay_tensor)? + noise_a)?;
    self.tensor_b = ((&self.tensor_b * &decay_tensor)? + noise_b)?;

        // Inject asymmetric divergence so tensors do not collapse to identical states.
        if self.divergence_strength > 0.0 {
            let shape = self.tensor_a.shape();
            let mod_a = Tensor::full(((self.time*0.7).sin()*0.5+0.5)*self.divergence_strength, shape, &self.device)?;
            let mod_b = Tensor::full(((self.time*1.1).cos()*0.5+0.5)*self.divergence_strength, shape, &self.device)?;
            self.tensor_a = (&self.tensor_a * (&mod_a + 1.0)?)?;
            self.tensor_b = (&self.tensor_b * (&mod_b + 1.0)?)?;
        }

        Ok(())
    }
    
    fn blend_tensors(&self, a: &Tensor, b: &Tensor, strength: f32) -> Result<Tensor> {
    let data_a = a.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
    let data_b = b.squeeze(0)?.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        
        let blended: Vec<f32> = data_a.iter().zip(data_b.iter())
            .map(|(&va, &vb)| va * (1.0 - strength) + vb * strength)
            .collect();
        
        Tensor::from_vec(blended, &[1, 1, TENSOR_HEIGHT, TENSOR_WIDTH], &self.device)
    }
    
    fn tensor_to_pixels(&self, tensor: &Tensor, x_offset: usize) -> Result<Vec<u32>> {
    let squeezed = tensor.squeeze(0)?.squeeze(0)?;
    let data = self.debug_flat_vec(&squeezed, "tensor_to_pixels.tensor")?;
        let mut pixels = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];
        
        let scale_x = DISPLAY_WIDTH as f32 / TENSOR_WIDTH as f32;
        let scale_y = WINDOW_HEIGHT as f32 / TENSOR_HEIGHT as f32;
        
        for y in 0..WINDOW_HEIGHT {
            for x in 0..DISPLAY_WIDTH {
                let tensor_x = (x as f32 / scale_x) as usize;
                let tensor_y = (y as f32 / scale_y) as usize;
                
                if tensor_x < TENSOR_WIDTH && tensor_y < TENSOR_HEIGHT {
                    let value = data[tensor_y * TENSOR_WIDTH + tensor_x];
                    let v_clamped = value.clamp(0.0,1.0);
                    let intensity = (v_clamped * 255.0) as u8;
                    let is_left = x_offset == 0;

                    // Distinct palettes per tensor and mode to make differences clearer.
                    let color = match self.mode {
                        ProcessingMode::DirectCoupling => {
                            if is_left { // cool
                                let r = intensity / 3; let g = intensity / 2; let b = intensity; (0xFF000000)|((r as u32)<<16)|((g as u32)<<8)|(b as u32)
                            } else { // warm
                                let r = intensity; let g = (intensity as f32*0.6) as u8; let b = intensity/4; (0xFF000000)|((r as u32)<<16)|((g as u32)<<8)|(b as u32)
                            }
                        }
                        ProcessingMode::CrossCoupling => {
                            let base_h = if is_left { 30.0 } else { 210.0 } + v_clamped*70.0;
                            let (r,g,b) = Self::hsv_to_rgb(base_h % 360.0, 0.85, v_clamped);
                            (0xFF000000)|((r as u32)<<16)|((g as u32)<<8)|(b as u32)
                        }
                        ProcessingMode::Interference => {
                            let phase = (self.time*0.9 + if is_left {0.0}else{1.2}).sin()*0.5+0.5;
                            let (r,g,b) = Self::hsv_to_rgb(phase*320.0, 1.0, v_clamped);
                            (0xFF000000)|((r as u32)<<16)|((g as u32)<<8)|(b as u32)
                        }
                        ProcessingMode::Convolution => {
                            let edge = (v_clamped*6.0).fract();
                            let (r,g,b) = if is_left { (intensity, (edge*255.0) as u8, 180u8) } else { ((edge*255.0) as u8, 40u8, intensity) };
                            (0xFF000000)|((r as u32)<<16)|((g as u32)<<8)|(b as u32)
                        }
                        ProcessingMode::FFTCoupling => {
                            let hue_shift = if is_left { 0.0 } else { 140.0 };
                            let hue = (v_clamped * 360.0 + hue_shift) % 360.0;
                            let (r,g,b) = Self::hsv_to_rgb(hue, 1.0, v_clamped);
                            (0xFF000000)|((r as u32)<<16)|((g as u32)<<8)|(b as u32)
                        }
                    };
                    
                    let pixel_x = x + x_offset;
                    if pixel_x < WINDOW_WIDTH {
                        pixels[y * WINDOW_WIDTH + pixel_x] = color;
                    }
                }
            }
        }
        
        Ok(pixels)
    }
    
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        
        (
            ((r + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((b + m) * 255.0) as u8,
        )
    }
    
    fn handle_input(&mut self) {
        // Mode selection
    if self.window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) { self.mode = ProcessingMode::DirectCoupling; }
    if self.window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) { self.mode = ProcessingMode::CrossCoupling; }
    if self.window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) { self.mode = ProcessingMode::Interference; }
    if self.window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) { self.mode = ProcessingMode::Convolution; }
    if self.window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) { self.mode = ProcessingMode::FFTCoupling; }
        
        // Filter selection
    if self.window.is_key_pressed(Key::F1, minifb::KeyRepeat::No) { self.filter_type = FilterType::None; }
    if self.window.is_key_pressed(Key::F2, minifb::KeyRepeat::No) { self.filter_type = FilterType::Gaussian; }
    if self.window.is_key_pressed(Key::F3, minifb::KeyRepeat::No) { self.filter_type = FilterType::Sobel; }
    if self.window.is_key_pressed(Key::F4, minifb::KeyRepeat::No) { self.filter_type = FilterType::Laplacian; }
    if self.window.is_key_pressed(Key::F5, minifb::KeyRepeat::No) { self.filter_type = FilterType::Emboss; }
        
        // Controls
    if self.window.is_key_down(Key::W) { self.coupling_strength = (self.coupling_strength + 0.01).min(1.0); }
    if self.window.is_key_down(Key::S) { self.coupling_strength = (self.coupling_strength - 0.01).max(0.0); }
    if self.window.is_key_down(Key::A) { self.rotation_speed -= 0.01; }
    if self.window.is_key_down(Key::D) { self.rotation_speed += 0.01; }
        
        // Reset
        if self.window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            self.tensor_a = Self::create_initial_pattern(&self.device, 0).unwrap();
            self.tensor_b = Self::create_initial_pattern(&self.device, 1).unwrap();
            self.time = 0.0;
            println!("[status] Reset tensors and time -> coupling {:.2} rotation {:.2} noise {} mode {:?} filter {:?}",
                self.coupling_strength, self.rotation_speed, if self.noise_level>0.0 {"on"} else {"off"}, self.mode, self.filter_type);
        }
        
        // Noise control
        if self.window.is_key_pressed(Key::N, minifb::KeyRepeat::No) {
            self.noise_level = if self.noise_level > 0.0 { 0.0 } else { 0.01 };
            println!("[status] Noise {}", if self.noise_level>0.0 {"enabled"} else {"disabled"});
        }

        // Report changes succinctly
        if self.last_report_mode != Some(self.mode) {
            println!("[status] Mode -> {:?}", self.mode);
            self.last_report_mode = Some(self.mode);
        }
        if self.last_report_filter != Some(self.filter_type) {
            println!("[status] Filter -> {:?}", self.filter_type);
            self.last_report_filter = Some(self.filter_type);
        }
        if (self.coupling_strength - self.last_report_coupling).abs() > 0.05 {
            println!("[status] Coupling -> {:.2}", self.coupling_strength);
            self.last_report_coupling = self.coupling_strength;
        }
        if (self.rotation_speed - self.last_report_rotation).abs() > 0.05 {
            println!("[status] Rotation speed -> {:.2}", self.rotation_speed);
            self.last_report_rotation = self.rotation_speed;
        }
        let noise_on = self.noise_level > 0.0;
        if noise_on != self.last_report_noise {
            println!("[status] Noise {}", if noise_on {"on"} else {"off"});
            self.last_report_noise = noise_on;
        }
    }
    
    fn update_title(&mut self) {
        let title = format!(
            "🔄 Tensor Closed Loop - Mode: {:?} | Filter: {:?} | Coupling: {:.2} | Rotation: {:.2}",
            self.mode, self.filter_type, self.coupling_strength, self.rotation_speed
        );
        self.window.set_title(&title);
    }
    
    fn run(&mut self) -> Result<()> {
        let mut last_time = Instant::now();
        
        while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
            let now = Instant::now();
            let dt = now.duration_since(last_time).as_secs_f32();
            last_time = now;
            
            // Update time and phase
            self.time += dt;
            self.phase += dt * 2.0;
            
            // Handle input
            self.handle_input();
            self.update_title();
            
            // Update feedback loop
            self.update_feedback()?;
            
            // Render both tensors side by side
            let pixels_a = self.tensor_to_pixels(&self.tensor_a, 0)?;
            let pixels_b = self.tensor_to_pixels(&self.tensor_b, DISPLAY_WIDTH)?;
            
            // Combine both displays
            let mut combined_pixels = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];
            for i in 0..pixels_a.len() {
                if pixels_a[i] != 0 {
                    combined_pixels[i] = pixels_a[i];
                }
                if pixels_b[i] != 0 {
                    combined_pixels[i] = pixels_b[i];
                }
            }
            
            // Draw separator line
            for y in 0..WINDOW_HEIGHT {
                let x = DISPLAY_WIDTH;
                if x < WINDOW_WIDTH {
                    combined_pixels[y * WINDOW_WIDTH + x] = 0xFFFFFFFF;
                }
            }
            
            // Update display
            self.window
                .update_with_buffer(&combined_pixels, WINDOW_WIDTH, WINDOW_HEIGHT)
                .unwrap();
                
            // Limit framerate
            std::thread::sleep(Duration::from_millis(16)); // ~60 FPS
        }
        
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("🔄 Starting Candle Tensor Closed Loop Visualizer...");
    println!();
    println!("🎮 CONTROLS:");
    println!("  [1-5]     - Processing modes (Direct, Cross, Interference, Convolution, FFT)");
    println!("  [F1-F5]   - Filters (None, Gaussian, Sobel, Laplacian, Emboss)");
    println!("  [W/S]     - Coupling strength ↑/↓");
    println!("  [A/D]     - Rotation speed ←/→");
    println!("  [R]       - Reset tensors");
    println!("  [N]       - Toggle noise");
    println!("  [ESC]     - Exit");
    println!();
    println!("📺 Two 2D tensors in closed loop - dynamic coupling system!");
    println!("🚀 Starting in 3 seconds...");
    
    std::thread::sleep(Duration::from_secs(3));
    
    let mut visualizer = TensorClosedLoopViz::new()?;
    visualizer.run()?;
    
    println!("👋 Tensor Closed Loop finished!");
    Ok(())
}
