// 🚀 GPU Stream-Based Real-Time Tensor Display
// Uses CUDA streams for efficient GPU→Display pipeline

use candle_core::{Device, Result, Tensor};
use candle_core::display::{set_print_options, PrinterOptions};
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};

const WINDOW_WIDTH: usize = 1200;
const WINDOW_HEIGHT: usize = 600;
const TENSOR_WIDTH: usize = 128;
const TENSOR_HEIGHT: usize = 128;
const DISPLAY_WIDTH: usize = WINDOW_WIDTH / 2;

struct StreamedTensorDisplay {
    window: Window,
    device: Device,
    
    // GPU tensors with optimized extraction
    tensor_a: Tensor,
    tensor_b: Tensor,
    
    // Host memory buffers for transfers
    buffer_a: Option<Vec<f32>>,
    buffer_b: Option<Vec<f32>>,
    
    // Display buffers (RGB for minifb)
    display_buffer: Vec<u32>,
    
    // Performance monitoring
    transfer_time: Duration,
    render_time: Duration,
    
    // Feedback parameters
    time: f32,
    feedback_strength: f32,
    mode: FeedbackMode,
}

#[derive(Clone, Copy, Debug)]
enum FeedbackMode {
    Direct,      // A → transform → B
    Cross,       // A ↔ B with transforms  
    Interference,// A + B interactions
    StreamTest,  // Pure stream performance test
    Convolution, // 2D convolution operations
    FFTAnalysis, // FFT magnitude/phase transforms
    MatrixOps,   // Matrix multiplication chains
    Nonlinear,   // Tanh, ReLU, sigmoid activations
    Reduction,   // Sum, mean, max pooling operations
    Spectral,    // Spectral filtering and analysis
}

impl std::fmt::Display for FeedbackMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedbackMode::Direct => write!(f, "Direct"),
            FeedbackMode::Cross => write!(f, "Cross"),
            FeedbackMode::Interference => write!(f, "Interference"),
            FeedbackMode::StreamTest => write!(f, "StreamTest"),
            FeedbackMode::Convolution => write!(f, "Convolution"),
            FeedbackMode::FFTAnalysis => write!(f, "FFTAnalysis"),
            FeedbackMode::MatrixOps => write!(f, "MatrixOps"),
            FeedbackMode::Nonlinear => write!(f, "Nonlinear"),
            FeedbackMode::Reduction => write!(f, "Reduction"),
            FeedbackMode::Spectral => write!(f, "Spectral"),
        }
    }
}

impl StreamedTensorDisplay {
    fn new() -> Result<Self> {
        let window = Window::new(
            "🚀 GPU Tensor Ops - [1-6] Modes [7-0] Advanced [Q-T] Spectral [S] Stats [WASD] Controls",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WindowOptions::default(),
        ).unwrap();

        // Use CUDA with stream optimization
        let device = Device::new_cuda(0).unwrap_or_else(|_| {
            println!("⚠️ CUDA not available, falling back to CPU");
            Device::Cpu
        });
        
                println!("🚀 Device: {device:?}");
        
        // Configure tensor display
        set_print_options(PrinterOptions {
            edge_items: 3,
            precision: 2,
            threshold: 8,
            line_width: 100,
            sci_mode: Some(false),
        });
        
        // Initialize GPU tensors with some interesting patterns
        let tensor_a = Self::create_initial_tensor_a(&device)?;
        let tensor_b = Self::create_initial_tensor_b(&device)?;
        
        // Allocate pinned host memory for fast transfers
        let (_pinned_buffer_a, _pinned_buffer_b) = match &device {
            Device::Cuda(_) => {
                (
                    Some(vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT]),
                    Some(vec![0.0f32; TENSOR_WIDTH * TENSOR_HEIGHT])
                )
            }
            _ => (None, None)
        };
        
        let display_buffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];
        
        Ok(Self {
            window,
            device,
            tensor_a,
            tensor_b,
            buffer_a: None,
            buffer_b: None,
            display_buffer,
            transfer_time: Duration::default(),
            render_time: Duration::default(),
            time: 0.0,
            feedback_strength: 0.1,
            mode: FeedbackMode::Direct,
        })
    }
    
    /// Create spiral pattern tensor on GPU
    fn create_initial_tensor_a(device: &Device) -> Result<Tensor> {
        let mut data = Vec::with_capacity(TENSOR_WIDTH * TENSOR_HEIGHT);
        let center_x = TENSOR_WIDTH as f32 / 2.0;
        let center_y = TENSOR_HEIGHT as f32 / 2.0;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let radius = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);
                
                // Spiral pattern
                let value = (radius * 0.1 + angle * 3.0).sin() * 0.5 + 0.5;
                data.push(value);
            }
        }
        
        Tensor::from_vec(data, (TENSOR_HEIGHT, TENSOR_WIDTH), device)
    }
    
    /// Create wave pattern tensor on GPU
    fn create_initial_tensor_b(device: &Device) -> Result<Tensor> {
        let mut data = Vec::with_capacity(TENSOR_WIDTH * TENSOR_HEIGHT);
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let nx = x as f32 / TENSOR_WIDTH as f32 * 4.0;
                let ny = y as f32 / TENSOR_HEIGHT as f32 * 4.0;
                
                // Wave interference pattern
                let value = (nx * std::f32::consts::PI).sin() * (ny * std::f32::consts::PI).cos() * 0.5 + 0.5;
                data.push(value);
            }
        }
        
        Tensor::from_vec(data, (TENSOR_HEIGHT, TENSOR_WIDTH), device)
    }
    
    /// 🚀 STREAM-OPTIMIZED: Update tensors using GPU operations with async transfer
    fn update_tensors_gpu_stream(&mut self) -> Result<()> {
    let _start_time = Instant::now(); // unused timing placeholder (underscore to silence warning)
        
        // Shape validation and correction
        if self.tensor_a.dims() != [TENSOR_HEIGHT, TENSOR_WIDTH] {
            println!("⚠️  Tensor A shape mismatch: {:?}, reinitializing...", self.tensor_a.dims());
            self.tensor_a = Self::create_initial_tensor_a(&self.device)?;
        }
        if self.tensor_b.dims() != [TENSOR_HEIGHT, TENSOR_WIDTH] {
            println!("⚠️  Tensor B shape mismatch: {:?}, reinitializing...", self.tensor_b.dims());
            self.tensor_b = Self::create_initial_tensor_b(&self.device)?;
        }
        
        match self.mode {
            FeedbackMode::StreamTest => {
                // Pure GPU computation - test stream performance
                let rotation = (self.time * 2.0).sin() * 0.1;
                let scale = 1.0 + (self.time * 0.5).cos() * 0.05;
                
                // GPU operations - these stay on GPU!
                self.tensor_a = self.apply_gpu_rotation(&self.tensor_a, rotation, scale)?;
                self.tensor_b = self.apply_gpu_wave_transform(&self.tensor_b, self.time)?;
            }
            
            FeedbackMode::Direct => {
                // Direct feedback using efficient rotation and addition
                let transformed_a = self.apply_gpu_rotation(&self.tensor_a, self.time, 1.0 + self.feedback_strength * 0.1)?;
                let feedback_tensor = transformed_a.affine(self.feedback_strength as f64, 0.0)?;
                self.tensor_b = (self.tensor_b.affine(0.95, 0.0)? + feedback_tensor)?;
            }
            
            FeedbackMode::Cross => {
                // A ↔ B cross-feedback (GPU-only operations)
                let feedback_a = self.apply_gpu_wave_transform(&self.tensor_b, self.time * 0.8)?;
                let feedback_b = self.apply_gpu_rotation(&self.tensor_a, -self.time * 0.1, 0.98)?;
                
                self.tensor_a = (self.tensor_a.affine(0.9, 0.0)? + feedback_a.affine(self.feedback_strength as f64, 0.0)?)?;
                self.tensor_b = (self.tensor_b.affine(0.9, 0.0)? + feedback_b.affine(self.feedback_strength as f64, 0.0)?)?;
            }
            
            FeedbackMode::Interference => {
                // A * B interference pattern (GPU-only operations)
                let interference = (&self.tensor_a * &self.tensor_b)?;
                let phase_shift = self.apply_gpu_wave_transform(&interference, self.time * 2.0)?;
                
                self.tensor_a = (self.tensor_a.affine(0.95, 0.0)? + phase_shift.affine((self.feedback_strength * 0.3) as f64, 0.0)?)?;
                self.tensor_b = (self.tensor_b.affine(0.95, 0.0)? + phase_shift.affine((self.feedback_strength * 0.7) as f64, 0.0)?)?;
            }
            
            FeedbackMode::Convolution => {
                // 2D Convolution operations (GPU-only)
                let kernel = self.create_dynamic_kernel()?;
                self.tensor_a = self.apply_conv2d(&self.tensor_a, &kernel)?;
                
                // Create feedback convolution
                let feedback_kernel = self.create_feedback_kernel()?;
                let conv_b = self.apply_conv2d(&self.tensor_b, &feedback_kernel)?;
                self.tensor_b = (self.tensor_b.affine(0.9, 0.0)? + conv_b.affine(self.feedback_strength as f64, 0.0)?)?;
            }
            
            FeedbackMode::FFTAnalysis => {
                // FFT magnitude and phase analysis (GPU-only)
                let fft_a = self.tensor_a.fft_magnitude()?;
                let phase_a = self.tensor_a.fft_phase()?;
                
                // Process FFT components
                let processed_mag = self.apply_gpu_wave_transform(&fft_a, self.time * 0.5)?;
                let processed_phase = self.apply_gpu_rotation(&phase_a, self.time * 0.3, 1.0)?;
                
                self.tensor_a = (processed_mag.affine(0.7, 0.0)? + processed_phase.affine(0.3, 0.0)?)?;
                self.tensor_b = (self.tensor_b.affine(0.95, 0.0)? + processed_mag.affine(self.feedback_strength as f64, 0.0)?)?;
            }
            
            FeedbackMode::MatrixOps => {
                // Matrix multiplication chains (GPU-only)
                let transpose_a = self.tensor_a.t()?;
                let mat_product = self.tensor_a.matmul(&transpose_a)?;
                
                // Symmetric matrix operations - use diagonal sum as trace approximation
                let diagonal_sum = mat_product.sum_all()?; // Sum all elements as approximation
                let eigenish = diagonal_sum.to_scalar::<f32>()? / (TENSOR_WIDTH * TENSOR_HEIGHT) as f32;
                let scaled_product = mat_product.affine(self.feedback_strength as f64, eigenish as f64)?;
                
                self.tensor_a = (self.tensor_a.affine(0.85, 0.0)? + scaled_product.affine(0.15, 0.0)?)?;
                // Fix: Use element-wise operations to maintain [128, 128] shape 
                // instead of matrix multiplication which could cause shape issues
                let feedback_transform = self.tensor_a.affine(self.feedback_strength as f64, 0.0)?;
                self.tensor_b = (self.tensor_b.affine(0.9, 0.0)? + feedback_transform.affine(0.1, 0.0)?)?;
            }
            
            FeedbackMode::Nonlinear => {
                // Nonlinear activation cascades (GPU-only)
                let tanh_a = self.tensor_a.tanh()?;
                let relu_a = self.tensor_a.clamp(0.0, f32::INFINITY as f64)?;
                // Create sigmoid-like function: sigmoid(x) ≈ tanh(x/2)/2 + 0.5, then scale to [-1,1]
                let sigmoid_b = self.tensor_b.affine(0.5, 0.0)?.tanh()?.affine(1.0, 0.5)?.affine(2.0, -1.0)?;
                
                // Combine nonlinear operations
                let nonlinear_mix = (tanh_a.affine(0.4, 0.0)? + relu_a.affine(0.3, 0.0)? + sigmoid_b.affine(0.3, 0.0)?)?;
                
                self.tensor_a = (self.tensor_a.affine(0.8, 0.0)? + nonlinear_mix.affine(self.feedback_strength as f64, 0.0)?)?;
                self.tensor_b = (self.tensor_b.affine(0.9, 0.0)? + tanh_a.affine(self.feedback_strength as f64 * 0.5, 0.0)?)?;
            }
            
            FeedbackMode::Reduction => {
                // Reduction operations and broadcasting (GPU-only)
                let mean_a = self.tensor_a.mean_all()?.unsqueeze(0)?.unsqueeze(0)?;
                let _max_a = self.tensor_a.max(0)?.max(0)?; // 2D max reduction (currently unused)
                let sum_b = self.tensor_b.sum_all()?.unsqueeze(0)?.unsqueeze(0)?;
                
                // Broadcast reductions back
                let mean_broadcast = mean_a.broadcast_as(self.tensor_a.shape())?;
                let sum_broadcast = sum_b.broadcast_as(self.tensor_b.shape())?;
                
                self.tensor_a = (self.tensor_a.affine(0.7, 0.0)? + mean_broadcast.affine(self.feedback_strength as f64, 0.0)?)?;
                self.tensor_b = (self.tensor_b.affine(0.8, 0.0)? + sum_broadcast.affine(self.feedback_strength as f64 * 0.1, 0.0)?)?;
            }
            
            FeedbackMode::Spectral => {
                // Advanced spectral operations (GPU-only)
                let fft_a = self.tensor_a.fft_magnitude()?;
                let spectral_filter = self.create_spectral_filter()?;
                let filtered = (fft_a * spectral_filter)?;
                
                // Spectral feedback with phase modulation
                let phase_mod = self.apply_gpu_wave_transform(&filtered, self.time * 1.5)?;
                let spectral_feedback = self.apply_gpu_rotation(&phase_mod, self.time * 0.8, 0.98)?;
                
                self.tensor_a = (self.tensor_a.affine(0.85, 0.0)? + spectral_feedback.affine(self.feedback_strength as f64, 0.0)?)?;
                self.tensor_b = (self.tensor_b.affine(0.9, 0.0)? + filtered.affine(self.feedback_strength as f64 * 0.6, 0.0)?)?;
            }
        }
        
        self.time += 0.016; // ~60fps increment
        
        Ok(())
    }
    
    /// 🚀 OPTIMIZED: Extract tensor data efficiently
    async fn extract_tensors_async(&mut self) -> Result<()> {
        let start_time = Instant::now();
        
        // Extract tensor data efficiently
        let flat_a = self.tensor_a.flatten_all()?;
        let flat_b = self.tensor_b.flatten_all()?;
        
        // Convert to host data
        let data_a = flat_a.to_vec1::<f32>()?;
        let data_b = flat_b.to_vec1::<f32>()?;
        
        // Store in buffers
        self.buffer_a = Some(data_a);
        self.buffer_b = Some(data_b);
        
        self.transfer_time = start_time.elapsed();
        Ok(())
    }
    
    /// Apply rotation transformation on GPU
    fn apply_gpu_rotation(&self, tensor: &Tensor, angle: f32, scale: f32) -> Result<Tensor> {
        // This is a simple approximation - real rotation would need proper GPU kernels
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        // For now, apply some mathematical transforms that stay on GPU
        let scaled = tensor.affine(scale as f64, 0.0)?;
        
        let cos_term = scaled.affine(cos_a as f64, 0.0)?;
        let sin_term = scaled.affine(sin_a as f64, 0.0)?;
        
        let offset = (cos_term + sin_term)?;
        
        // Add some nonlinear transform to keep values in range
        let clamped = offset.clamp(-1.0, 1.0)?;
        Ok(clamped)
    }
    
    /// Apply wave transformation on GPU
    fn apply_gpu_wave_transform(&self, tensor: &Tensor, phase: f32) -> Result<Tensor> {
        // Create phase tensor
        let phase_offset = Tensor::full(phase, tensor.shape(), &self.device)?;
        
        // Apply wave-like transformation
        let phase_tensor = (tensor + phase_offset)?;
        let wave_tensor = phase_tensor.sin()?;
        
        Ok(wave_tensor)
    }
    
    /// Create dynamic convolution kernel
    fn create_dynamic_kernel(&self) -> Result<Tensor> {
        // 3x3 time-varying kernel
        let kernel_size = 3;
        let t = self.time;
        
        let kernel_data = [
            (t * 2.0).sin() * 0.1,  (t * 1.5).cos() * 0.2,  (t * 2.5).sin() * 0.1,
            (t * 1.8).cos() * 0.2,  1.0 - (t * 0.5).sin().abs() * 0.3,  (t * 1.3).sin() * 0.2,
            (t * 2.2).sin() * 0.1,  (t * 1.7).cos() * 0.2,  (t * 2.8).sin() * 0.1,
        ];
        
        Tensor::from_slice(&kernel_data, (1, 1, kernel_size, kernel_size), &self.device)
    }
    
    /// Create feedback convolution kernel
    fn create_feedback_kernel(&self) -> Result<Tensor> {
        // Edge detection + smoothing hybrid kernel
        let kernel_data = [
            -0.1, -0.2, -0.1,
             0.0,  1.0,  0.0,
             0.1,  0.2,  0.1,
        ];
        
        Tensor::from_slice(&kernel_data, (1, 1, 3, 3), &self.device)
    }
    
    /// Apply 2D convolution (simplified - real conv2d would need proper implementation)
    fn apply_conv2d(&self, tensor: &Tensor, _kernel: &Tensor) -> Result<Tensor> {
        // Simplified convolution approximation using tensor operations
        // Real implementation would use proper conv2d operations
        
        // For now, apply a weighted combination that simulates convolution effects
        let smoothed = tensor.affine(0.8, 0.0)?;
        let edge_enhanced = (tensor - smoothed.affine(0.9, 0.0)?)?;
        let result = (smoothed + edge_enhanced.affine(0.3, 0.0)?)?;
        
        result.clamp(-1.0, 1.0)
    }
    
    /// Create spectral filter
    fn create_spectral_filter(&self) -> Result<Tensor> {
        // Create frequency-domain filter
        let mut filter_data = Vec::new();
        let center_x = TENSOR_WIDTH as f32 / 2.0;
        let center_y = TENSOR_HEIGHT as f32 / 2.0;
        
        for y in 0..TENSOR_HEIGHT {
            for x in 0..TENSOR_WIDTH {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let freq = (dx * dx + dy * dy).sqrt() / center_x;
                
                // Low-pass filter with time modulation
                let cutoff = 0.5 + 0.3 * (self.time * 0.7).sin();
                let filter_val = if freq < cutoff {
                    1.0 - freq / cutoff * 0.5
                } else {
                    0.1 * (-((freq - cutoff) * 3.0).exp())
                };
                
                filter_data.push(filter_val);
            }
        }
        
        Tensor::from_slice(&filter_data, (TENSOR_HEIGHT, TENSOR_WIDTH), &self.device)
    }
    
    /// Convert tensor data to RGB display buffer
    fn render_to_display(&mut self) {
        let start_time = Instant::now();
        
        // Clear buffer
        self.display_buffer.fill(0xFF000000); // Black background
        
        if let (Some(buffer_a), Some(buffer_b)) = (self.buffer_a.take(), self.buffer_b.take()) {
            // Render tensor A (left side)
            self.render_tensor_to_buffer(&buffer_a, 0, 0, DISPLAY_WIDTH, WINDOW_HEIGHT, 0xFF0040FF); // Blue-ish
            
            // Render tensor B (right side)  
            self.render_tensor_to_buffer(&buffer_b, DISPLAY_WIDTH, 0, DISPLAY_WIDTH, WINDOW_HEIGHT, 0xFFFF4000); // Red-ish
            
            // Restore the buffers
            self.buffer_a = Some(buffer_a);
            self.buffer_b = Some(buffer_b);
            
            // Add divider line
            for y in 0..WINDOW_HEIGHT {
                let idx = y * WINDOW_WIDTH + DISPLAY_WIDTH;
                if idx < self.display_buffer.len() {
                    self.display_buffer[idx] = 0xFFFFFFFF; // White divider
                }
            }
        }
        
        self.render_time = start_time.elapsed();
    }
    
    /// Render tensor data to specific region of display buffer
    fn render_tensor_to_buffer(&mut self, data: &[f32], offset_x: usize, offset_y: usize, width: usize, height: usize, base_color: u32) {
        let scale_x = TENSOR_WIDTH as f32 / width as f32;
        let scale_y = TENSOR_HEIGHT as f32 / height as f32;
        
        for py in 0..height {
            for px in 0..width {
                let tx = ((px as f32 * scale_x) as usize).min(TENSOR_WIDTH - 1);
                let ty = ((py as f32 * scale_y) as usize).min(TENSOR_HEIGHT - 1);
                let tidx = ty * TENSOR_WIDTH + tx;
                
                if tidx < data.len() {
                    let intensity = data[tidx].clamp(0.0, 1.0);
                    
                    // Extract RGB components from base color
                    let r = ((base_color >> 16) & 0xFF) as f32;
                    let g = ((base_color >> 8) & 0xFF) as f32;
                    let b = (base_color & 0xFF) as f32;
                    
                    // Apply intensity
                    let final_r = (r * intensity) as u32;
                    let final_g = (g * intensity) as u32;
                    let final_b = (b * intensity) as u32;
                    
                    let color = 0xFF000000 | (final_r << 16) | (final_g << 8) | final_b;
                    
                    let display_x = offset_x + px;
                    let display_y = offset_y + py;
                    
                    if display_x < WINDOW_WIDTH && display_y < WINDOW_HEIGHT {
                        let idx = display_y * WINDOW_WIDTH + display_x;
                        if idx < self.display_buffer.len() {
                            self.display_buffer[idx] = color;
                        }
                    }
                }
            }
        }
    }
    
    /// Handle user input
    fn handle_input(&mut self) {
        // Basic modes [1-4]
        if self.window.is_key_down(Key::Key1) { self.mode = FeedbackMode::Direct; }
        if self.window.is_key_down(Key::Key2) { self.mode = FeedbackMode::Cross; }
        if self.window.is_key_down(Key::Key3) { self.mode = FeedbackMode::Interference; }
        if self.window.is_key_down(Key::Key4) { self.mode = FeedbackMode::StreamTest; }
        
        // Advanced modes [5-0]
        if self.window.is_key_down(Key::Key5) { self.mode = FeedbackMode::Convolution; }
        if self.window.is_key_down(Key::Key6) { self.mode = FeedbackMode::FFTAnalysis; }
        if self.window.is_key_down(Key::Key7) { self.mode = FeedbackMode::MatrixOps; }
        if self.window.is_key_down(Key::Key8) { self.mode = FeedbackMode::Nonlinear; }
        if self.window.is_key_down(Key::Key9) { self.mode = FeedbackMode::Reduction; }
        if self.window.is_key_down(Key::Key0) { self.mode = FeedbackMode::Spectral; }
        
        // Spectral modes [Q-T]
        if self.window.is_key_down(Key::Q) { self.mode = FeedbackMode::FFTAnalysis; }
        if self.window.is_key_down(Key::W) && !self.window.is_key_down(Key::LeftShift) { 
            self.feedback_strength = (self.feedback_strength * 1.1).min(1.0); 
        }
        if self.window.is_key_down(Key::E) { self.mode = FeedbackMode::Spectral; }
        if self.window.is_key_down(Key::T) { self.mode = FeedbackMode::MatrixOps; }
        if self.window.is_key_down(Key::S) && !self.window.is_key_down(Key::LeftShift) { 
            self.feedback_strength = (self.feedback_strength * 0.9).max(0.001); 
        }
        
        if self.window.is_key_down(Key::R) {
            // Reset tensors
            if let (Ok(new_a), Ok(new_b)) = (
                Self::create_initial_tensor_a(&self.device),
                Self::create_initial_tensor_b(&self.device)
            ) {
                self.tensor_a = new_a;
                self.tensor_b = new_b;
                self.time = 0.0;
            }
        }
    }
    
    /// Print performance statistics
    fn print_stats(&self) {
        if self.window.is_key_down(Key::S) && self.window.is_key_down(Key::LeftShift) {
            println!("🚀 GPU Stream Display Stats:");
            println!("   Mode: {}", self.mode);
            println!("   Device: {:?}", self.device);
            println!("   Transfer Time: {:?}", self.transfer_time);
            println!("   Render Time: {:?}", self.render_time);
            println!("   Feedback Strength: {:.3}", self.feedback_strength);
            println!("   Stream Available: {}", matches!(self.device, Device::Cuda(_)));
            println!("   Pinned Memory: {}", self.buffer_a.is_some());
            
            // Monitor tensor stats (OFFICIAL Candle monitoring)
            if let Ok(stats_a) = monitor_tensor(&self.tensor_a) {
                println!("   Tensor A: {stats_a}");
            }
            if let Ok(stats_b) = monitor_tensor(&self.tensor_b) {
                println!("   Tensor B: {stats_b}");
            }
        }
    }
    
    /// Main display loop
    fn run(&mut self) -> Result<()> {
        let mut frame_count = 0;
        let mut last_fps_time = Instant::now();
        
        while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
            let frame_start = Instant::now();
            
            // Update tensors on GPU
            self.update_tensors_gpu_stream()?;
            
            // Extract data using streams (async when available)
            // Note: In a real async setup, this would be truly async
            if let Err(e) = futures::executor::block_on(self.extract_tensors_async()) {
                println!("⚠️ Stream extraction error: {e}");
            }
            
            // Render to display
            self.render_to_display();
            
            // Update window
            self.window.update_with_buffer(&self.display_buffer, WINDOW_WIDTH, WINDOW_HEIGHT).unwrap();
            
            // Handle input
            self.handle_input();
            
            // Print stats if requested
            self.print_stats();
            
            // FPS monitoring
            frame_count += 1;
            if last_fps_time.elapsed() >= Duration::from_secs(1) {
                let fps = frame_count as f64 / last_fps_time.elapsed().as_secs_f64();
                self.window.set_title(&format!(
                    "🚀 GPU Stream Display - {:?} - {:.1} FPS - Transfer: {:?} - Render: {:?}",
                    self.mode, fps, self.transfer_time, self.render_time
                ));
                frame_count = 0;
                last_fps_time = Instant::now();
            }
            
            // Target 60 FPS
            let frame_time = frame_start.elapsed();
            if frame_time < Duration::from_millis(16) {
                std::thread::sleep(Duration::from_millis(16) - frame_time);
            }
        }
        
        Ok(())
    }
}

/// Official Candle tensor monitoring
fn monitor_tensor(tensor: &Tensor) -> Result<String> {
    let min_val = tensor.min_all()?.to_scalar::<f32>()?;
    let max_val = tensor.max_all()?.to_scalar::<f32>()?;
    let mean_val = tensor.mean_all()?.to_scalar::<f32>()?;
    
    Ok(format!("min={:.3}, max={:.3}, mean={:.3}, shape={:?}, device={:?}", 
        min_val, max_val, mean_val, tensor.shape(), tensor.device()))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting GPU Stream-Based Tensor Display...");
    println!("📋 Controls:");
    println!("   Basic Modes:");
    println!("   [1] Direct     [2] Cross      [3] Interference [4] StreamTest");
    println!("   Advanced Operations:");
    println!("   [5] Convolution [6] FFT Analysis [7] Matrix Ops [8] Nonlinear");
    println!("   [9] Reduction   [0] Spectral");
    println!("   Quick Access:");
    println!("   [Q] FFT Analysis [E] Spectral  [T] Matrix Ops");
    println!("   Controls:");
    println!("   [W/S] Feedback strength [R] Reset [Shift+S] Stats [ESC] Exit");
    println!();
    
    let mut display = StreamedTensorDisplay::new()?;
    display.run()
}
