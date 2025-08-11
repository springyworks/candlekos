// CPU FFT implementation using Intel MKL DFT or pure Rust fallback
use crate::{Result};
use crate::layout::Layout;

/// FFT configuration for CPU operations
#[derive(Debug, Clone, Copy)]
pub struct CpuFftConfig {
    pub forward: bool,    // true for forward FFT, false for inverse
    pub normalized: bool, // apply normalization factor
    pub real_input: bool, // real-to-complex FFT (vs complex-to-complex)
}

impl Default for CpuFftConfig {
    fn default() -> Self {
        Self {
            forward: true,
            normalized: true,
            real_input: false,
        }
    }
}

/// CPU FFT operation implementation
#[derive(Debug, Clone, Copy)]
pub struct CpuFft {
    pub config: CpuFftConfig,
    pub dim: usize, // dimension along which to perform FFT
}

impl CpuFft {
    pub fn new(config: CpuFftConfig, dim: usize) -> Self {
        Self { config, dim }
    }

    /// Execute FFT on CPU using available backend
    pub fn fft_f32(
        &self,
        input: &[f32],
        layout: &Layout,
    ) -> Result<Vec<f32>> {
        let shape = layout.shape();
        let dims = shape.dims();
        
        if self.dim >= dims.len() {
            return Err(crate::Error::DimOutOfRange {
                shape: shape.clone(),
                dim: self.dim as i32,
                op: "fft",
            }.bt());
        }

        let n = dims[self.dim];
        
        #[cfg(feature = "mkl")]
        {
            self.fft_mkl_f32(input, layout, n)
        }
        
        #[cfg(not(feature = "mkl"))]
        {
            self.fft_rust_f32(input, layout, n)
        }
    }

    #[cfg(feature = "mkl")]
    fn fft_mkl_f32(&self, input: &[f32], layout: &Layout, n: usize) -> Result<Vec<f32>> {
        use std::ffi::c_void;
        
        // Intel MKL DFT interface
        extern "C" {
            fn DftiCreateDescriptor(
                hand: *mut *mut c_void,
                precision: i32,
                domain: i32,
                dimension: i32,
                length: i32,
            ) -> i32;
            
            fn DftiSetValue(hand: *mut c_void, config: i32, value: *const c_void) -> i32;
            fn DftiCommitDescriptor(hand: *mut c_void) -> i32;
            fn DftiComputeForward(hand: *mut c_void, input: *mut c_void, output: *mut c_void) -> i32;
            fn DftiComputeBackward(hand: *mut c_void, input: *mut c_void, output: *mut c_void) -> i32;
            fn DftiFreeDescriptor(hand: *mut *mut c_void) -> i32;
        }

        const DFTI_SINGLE: i32 = 35; // Single precision
        const DFTI_REAL: i32 = 36;   // Real domain
        const DFTI_COMPLEX: i32 = 37; // Complex domain
        const DFTI_PLACEMENT: i32 = 11;
        const DFTI_NOT_INPLACE: i32 = 44;

        let mut descriptor: *mut c_void = std::ptr::null_mut();
        
        // Create descriptor
        let domain = if self.config.real_input { DFTI_REAL } else { DFTI_COMPLEX };
        let status = unsafe {
            DftiCreateDescriptor(
                &mut descriptor,
                DFTI_SINGLE,
                domain,
                1, // 1D FFT
                n as i32,
            )
        };
        
        if status != 0 {
            return Err(crate::Error::Msg("Failed to create MKL DFT descriptor".to_string()).bt());
        }

        // Configure out-of-place computation
        let not_inplace = DFTI_NOT_INPLACE;
        let status = unsafe {
            DftiSetValue(descriptor, DFTI_PLACEMENT, &not_inplace as *const i32 as *const c_void)
        };
        
        if status != 0 {
            unsafe { DftiFreeDescriptor(&mut descriptor) };
            return Err(crate::Error::Msg("Failed to configure MKL DFT".to_string()).bt());
        }

        // Commit the descriptor
        let status = unsafe { DftiCommitDescriptor(descriptor) };
        if status != 0 {
            unsafe { DftiFreeDescriptor(&mut descriptor) };
            return Err(crate::Error::Msg("Failed to commit MKL DFT descriptor".to_string()).bt());
        }

        // Prepare input and output buffers
        let total_size = layout.shape().elem_count();
        let mut output = vec![0.0f32; if self.config.real_input { 
            // For R2C, output size is (n/2+1)*2 for each FFT
            total_size / n * (n / 2 + 1) * 2
        } else { 
            // For C2C, same size
            total_size * 2 // Complex numbers have real and imaginary parts
        }];

        // Perform FFT
        let status = if self.config.forward {
            unsafe {
                DftiComputeForward(
                    descriptor,
                    input.as_ptr() as *mut c_void,
                    output.as_mut_ptr() as *mut c_void,
                )
            }
        } else {
            unsafe {
                DftiComputeBackward(
                    descriptor,
                    input.as_ptr() as *mut c_void,
                    output.as_mut_ptr() as *mut c_void,
                )
            }
        };

        // Clean up
        unsafe { DftiFreeDescriptor(&mut descriptor) };

        if status != 0 {
            return Err(crate::Error::Msg("MKL DFT computation failed".to_string()).bt());
        }

        // Apply normalization if requested
        if self.config.normalized {
            let norm_factor = if self.config.forward {
                1.0 / (n as f32).sqrt()
            } else {
                1.0 / (n as f32).sqrt()
            };
            
            for val in output.iter_mut() {
                *val *= norm_factor;
            }
        }

        Ok(output)
    }

    #[cfg(not(feature = "mkl"))]
    fn fft_rust_f32(&self, input: &[f32], layout: &Layout, n: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "fft")]
        {
            // Pure Rust FFT implementation using RustFFT
            use rustfft::{FftPlanner, num_complex::Complex};
            
            let mut planner = FftPlanner::new();
            let fft = if self.config.forward {
                planner.plan_fft_forward(n)
            } else {
                planner.plan_fft_inverse(n)
            };

            let total_size = layout.shape().elem_count();
            let batch_size = total_size / n;
            
            let mut output = Vec::with_capacity(total_size * 2);

            for batch in 0..batch_size {
                let start_idx = batch * n;
                let end_idx = start_idx + n;
                
                // Convert input to complex numbers
                let mut buffer: Vec<Complex<f32>> = if self.config.real_input {
                    input[start_idx..end_idx]
                        .iter()
                        .map(|&x| Complex::new(x, 0.0))
                        .collect()
                } else {
                    // Assume input is interleaved real/imaginary
                    input[start_idx * 2..end_idx * 2]
                        .chunks_exact(2)
                        .map(|chunk| Complex::new(chunk[0], chunk[1]))
                        .collect()
                };

                // Perform FFT
                fft.process(&mut buffer);

                // Apply normalization if requested
                if self.config.normalized {
                    let norm_factor = if self.config.forward {
                        1.0 / (n as f32).sqrt()
                    } else {
                        1.0 / (n as f32).sqrt()
                    };
                    
                    for val in buffer.iter_mut() {
                        *val *= norm_factor;
                    }
                }

                // Convert back to interleaved format
                for complex_val in buffer {
                    output.push(complex_val.re);
                    output.push(complex_val.im);
                }
            }

            Ok(output)
        }
        
        #[cfg(not(feature = "fft"))]
        {
            Err(crate::Error::Msg("FFT not available. Enable 'fft' feature for RustFFT fallback".to_string()).bt())
        }
    }

    /// Execute real-to-complex FFT
    pub fn rfft_f32(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>> {
        let mut config = self.config;
        config.real_input = true;
        let fft = CpuFft::new(config, self.dim);
        fft.fft_f32(input, layout)
    }

    /// Compute magnitude spectrum from complex FFT output
    pub fn magnitude_f32(&self, complex_output: &[f32]) -> Vec<f32> {
        complex_output
            .chunks_exact(2)
            .map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];
                (real * real + imag * imag).sqrt()
            })
            .collect()
    }

    /// Compute phase spectrum from complex FFT output
    pub fn phase_f32(&self, complex_output: &[f32]) -> Vec<f32> {
        complex_output
            .chunks_exact(2)
            .map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];
                imag.atan2(real)
            })
            .collect()
    }
}

/// 2D FFT implementation
impl CpuFft {
    /// Execute 2D FFT on the last two dimensions
    pub fn fft2_f32(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>> {
        let dims = layout.dims();
        
        if dims.len() < 2 {
            return Err(crate::Error::Msg("2D FFT requires at least 2 dimensions".to_string()).bt());
        }
        
        // First do FFT on the second-to-last dimension, then on the last dimension
        let intermediate = self.fft_along_axis(input, layout, dims.len() - 2)?;
        
        // Create new layout for the intermediate result
        let mut intermediate_dims = dims.to_vec();
        if self.config.real_input {
            intermediate_dims[dims.len() - 2] = (intermediate_dims[dims.len() - 2] / 2 + 1) * 2;
        }
        let intermediate_layout = Layout::contiguous(intermediate_dims);
        
        // Create a new FFT op for the second pass (now working with complex data)
        let second_pass_config = CpuFftConfig {
            forward: self.config.forward,
            normalized: self.config.normalized,
            real_input: false, // Second pass is always complex-to-complex
        };
        let second_pass_fft = CpuFft::new(second_pass_config, dims.len() - 1);
        
        second_pass_fft.fft_along_axis(&intermediate, &intermediate_layout, dims.len() - 1)
    }

    /// Execute FFT along a specific axis
    fn fft_along_axis(&self, input: &[f32], layout: &Layout, axis: usize) -> Result<Vec<f32>> {
        // Create a temporary FFT operator for this axis
        let axis_config = CpuFftConfig {
            forward: self.config.forward,
            normalized: self.config.normalized,
            real_input: self.config.real_input,
        };
        let axis_fft = CpuFft::new(axis_config, axis);
        axis_fft.fft_f32(input, layout)
    }
}

// Window functions for FFT preprocessing
impl CpuFft {
    /// Apply Hann window to input data
    pub fn apply_hann_window(&self, data: &mut [f32], window_size: usize) {
        for (i, val) in data.iter_mut().enumerate() {
            let n = (i % window_size) as f32;
            let factor = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / (window_size - 1) as f32).cos());
            *val *= factor;
        }
    }

    /// Apply Hamming window to input data
    pub fn apply_hamming_window(&self, data: &mut [f32], window_size: usize) {
        for (i, val) in data.iter_mut().enumerate() {
            let n = (i % window_size) as f32;
            let factor = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n / (window_size - 1) as f32).cos();
            *val *= factor;
        }
    }

    /// Apply Blackman window to input data
    pub fn apply_blackman_window(&self, data: &mut [f32], window_size: usize) {
        const A0: f32 = 0.42;
        const A1: f32 = 0.5;
        const A2: f32 = 0.08;
        
        for (i, val) in data.iter_mut().enumerate() {
            let n = (i % window_size) as f32;
            let arg = 2.0 * std::f32::consts::PI * n / (window_size - 1) as f32;
            let factor = A0 - A1 * arg.cos() + A2 * (2.0 * arg).cos();
            *val *= factor;
        }
    }
}

// FFT shift operations
impl CpuFft {
    /// FFT shift: move zero frequency to center
    pub fn fftshift(&self, data: &mut [f32], fft_size: usize) {
        let complex_size = fft_size / 2; // Assuming complex interleaved data
        for batch_start in (0..data.len()).step_by(fft_size) {
            let batch_end = (batch_start + fft_size).min(data.len());
            let batch = &mut data[batch_start..batch_end];
            
            // Rotate by half the FFT size
            batch.rotate_left(complex_size);
        }
    }

    /// Inverse FFT shift: undo fftshift
    pub fn ifftshift(&self, data: &mut [f32], fft_size: usize) {
        let complex_size = fft_size / 2;
        for batch_start in (0..data.len()).step_by(fft_size) {
            let batch_end = (batch_start + fft_size).min(data.len());
            let batch = &mut data[batch_start..batch_end];
            
            // Rotate by half the FFT size in the other direction
            batch.rotate_right(complex_size);
        }
    }
}

