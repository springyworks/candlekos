// CPU FFT implementation using Intel MKL DFT or pure Rust fallback
use crate::{CpuStorage, Result, WithDType};
use crate::layout::Layout;
use crate::shape::Shape;

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
        
        let mut output = if self.config.real_input {
            // For real input, convert to complex and perform C2C FFT
            Vec::with_capacity(total_size * 2)
        } else {
            Vec::with_capacity(total_size * 2)
        };

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

// Add rustfft dependency for pure Rust fallback
#[cfg(not(feature = "mkl"))]
mod rustfft_dep {
    // This will need to be added to Cargo.toml
    // rustfft = "6.0"
}
