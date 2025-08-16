//! CPU FFT implementation using RustFFT for signal processing operations.
//! Provides 1D, 2D, and multi-dimensional FFT operations with real-to-complex and complex-to-complex transforms.

// CPU FFT implementation using Intel MKL DFT or pure Rust fallback
use crate::Result;
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
    pub fn fft_f32(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>> {
        let shape = layout.shape();
        let dims = shape.dims();

        if self.dim >= dims.len() {
            return Err(crate::Error::DimOutOfRange {
                shape: shape.clone(),
                dim: self.dim as i32,
                op: "fft",
            }
            .bt());
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
            fn DftiComputeForward(
                hand: *mut c_void,
                input: *mut c_void,
                output: *mut c_void,
            ) -> i32;
            fn DftiComputeBackward(
                hand: *mut c_void,
                input: *mut c_void,
                output: *mut c_void,
            ) -> i32;
            fn DftiFreeDescriptor(hand: *mut *mut c_void) -> i32;
        }

        const DFTI_SINGLE: i32 = 35; // Single precision
        const DFTI_REAL: i32 = 36; // Real domain
        const DFTI_COMPLEX: i32 = 37; // Complex domain
        const DFTI_PLACEMENT: i32 = 11;
        const DFTI_NOT_INPLACE: i32 = 44;

        let mut descriptor: *mut c_void = std::ptr::null_mut();

        // Create descriptor
        let domain = if self.config.real_input {
            DFTI_REAL
        } else {
            DFTI_COMPLEX
        };
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
            DftiSetValue(
                descriptor,
                DFTI_PLACEMENT,
                &not_inplace as *const i32 as *const c_void,
            )
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
        let mut output = vec![
            0.0f32;
            if self.config.real_input {
                // For R2C, output size is (n/2+1)*2 for each FFT
                total_size / n * (n / 2 + 1) * 2
            } else {
                // For C2C, same size
                total_size * 2 // Complex numbers have real and imaginary parts
            }
        ];

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
        if self.config.real_input && self.config.forward {
            let shape = layout.shape();
            let dims = shape.dims();
            let total_size = shape.elem_count();
            let batch_size = total_size / n;
            let stride = layout.stride()[self.dim];
            println!(
                "[DEBUG] cpu_fft: n = {}, batch_size = {}, stride = {}, dims = {:?}, input.len() = {}",
                n,
                batch_size,
                stride,
                dims,
                input.len()
            );
        }
        #[cfg(feature = "fft")]
        {
            // FFT implementation using RustFFT
            use rustfft::{FftPlanner, num_complex::Complex};

            let mut planner = FftPlanner::new();
            let fft = if self.config.forward {
                planner.plan_fft_forward(n)
            } else {
                planner.plan_fft_inverse(n)
            };

            let shape = layout.shape();
            let dims = shape.dims();
            let total_size = shape.elem_count();

            // Calculate batch size correctly for the FFT dimension
            let batch_size = total_size / n;
            let stride = layout.stride()[self.dim];

            let mut output = Vec::new();

            // Handle real-to-complex vs complex-to-real FFT
            if self.config.real_input {
                if self.config.forward {
                    // Forward real-to-complex FFT: input size n, output size (n/2+1)*2
                    output.reserve(batch_size * (n / 2 + 1) * 2);

                    for batch in 0..batch_size {
                        // Calculate the starting position for this batch
                        let batch_start = self.calculate_batch_start(batch, dims, stride, n);

                        // Extract real input data with proper striding
                        let mut real_input: Vec<f32> = Vec::with_capacity(n);
                        for i in 0..n {
                            let idx = batch_start + i * stride;
                            if idx < input.len() {
                                real_input.push(input[idx]);
                            } else {
                                return Err(crate::Error::Msg(format!("Real input index out of bounds: trying to access {} in array of length {}", idx, input.len())).bt());
                            }
                        }

                        // Convert to complex and perform FFT
                        let mut buffer: Vec<Complex<f32>> =
                            real_input.iter().map(|&x| Complex::new(x, 0.0)).collect();

                        fft.process(&mut buffer);

                        // Apply normalization if requested
                        if self.config.normalized {
                            let norm_factor = 1.0 / (n as f32).sqrt();
                            for val in buffer.iter_mut() {
                                *val *= norm_factor;
                            }
                        }

                        // For real FFT, only output first (n/2+1) complex values (Hermitian symmetry)
                        let output_len = n / 2 + 1;
                        for i in 0..output_len {
                            output.push(buffer[i].re);
                            output.push(buffer[i].im);
                        }
                    }
                } else {
                    // Inverse complex-to-real FFT: input size (n/2+1)*2, output size n
                    // Use RustFFT's RealFftPlanner for inverse real FFT
                    let n = input.len() - 2;
                    let real_input_size = input.len();
                    let num_batches = 1; // Only batch size 1 supported for now
                    output.reserve(num_batches * n);

                    use realfft::RealFftPlanner;
                    let mut real_planner = RealFftPlanner::<f32>::new();
                    let irfft = real_planner.plan_fft_inverse(n);

                    for batch in 0..num_batches {
                        let batch_start = batch * real_input_size;
                        let input_slice = &input[batch_start..batch_start + real_input_size];
                        // Convert interleaved real/imag to Vec<Complex<f32>>
                        let mut spectrum = Vec::with_capacity(n / 2 + 1);
                        for i in 0..(n / 2 + 1) {
                            let re = input_slice[i * 2];
                            let im = input_slice[i * 2 + 1];
                            spectrum.push(Complex::new(re, im));
                        }
                        let mut out = vec![0.0f32; n];
                        irfft.process(&mut spectrum, &mut out).map_err(|e| {
                            crate::Error::Msg(format!("realfft irfft error: {}", e)).bt()
                        })?;
                        // Apply normalization if requested
                        if self.config.normalized {
                            let norm_factor = 1.0 / (n as f32).sqrt();
                            for val in out.iter_mut() {
                                *val *= norm_factor;
                            }
                        }
                        output.extend_from_slice(&out);
                        // End of irfft batch loop
                    }
                }
            } else {
                // Complex-to-complex FFT: output same size as input
                output.reserve(total_size);

                for batch in 0..batch_size {
                    // Extract complex input data with improved indexing logic
                    let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(n);

                    // Check if we have enough data for n complex numbers (2*n floats)
                    let expected_complex_size = batch_size * n * 2;
                    if input.len() != expected_complex_size {
                        // Input size doesn't match expected complex array size
                        // This might be a real array being treated as complex
                        if input.len() == batch_size * n {
                            // Real input being processed as complex - convert it
                            let real_batch_start = batch * n;
                            for i in 0..n {
                                let real_idx = real_batch_start + i;
                                if real_idx < input.len() {
                                    buffer.push(Complex::new(input[real_idx], 0.0));
                                } else {
                                    return Err(crate::Error::Msg(format!("Real-as-complex input index out of bounds: trying to access {} in array of length {}", real_idx, input.len())).bt());
                                }
                            }
                        } else {
                            return Err(crate::Error::Msg(format!("Input size mismatch: expected {} (complex) or {} (real) floats, got {}", expected_complex_size, batch_size * n, input.len())).bt());
                        }
                    } else {
                        // True complex input - use complex indexing
                        let complex_batch_start = batch * n * 2;
                        for i in 0..n {
                            let real_idx = complex_batch_start + i * 2;
                            let imag_idx = real_idx + 1;

                            if imag_idx < input.len() {
                                buffer.push(Complex::new(input[real_idx], input[imag_idx]));
                            } else {
                                return Err(crate::Error::Msg(format!("Complex input index out of bounds: trying to access {}..{} in array of length {} (batch {}, i {})", real_idx, imag_idx + 1, input.len(), batch, i)).bt());
                            }
                        }
                    }

                    // Perform FFT
                    fft.process(&mut buffer);

                    // Apply normalization if requested
                    if self.config.normalized {
                        let norm_factor = 1.0 / (n as f32).sqrt();
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
            }

            Ok(output)
        }

        #[cfg(not(feature = "fft"))]
        {
            Err(crate::Error::Msg(
                "FFT not available. Enable 'fft' feature for RustFFT fallback".to_string(),
            )
            .bt())
        }
    }

    /// Calculate the starting index for a batch in the input array
    #[allow(dead_code)]
    fn calculate_batch_start(
        &self,
        batch: usize,
        dims: &[usize],
        _stride: usize,
        n: usize,
    ) -> usize {
        if dims.len() == 1 {
            // 1D case: simple linear indexing
            batch * n
        } else {
            // Multi-dimensional case: calculate position considering the FFT dimension
            let mut batch_idx = batch;
            let mut start = 0;
            let mut remaining_dims = Vec::new();

            // Collect dimensions excluding the FFT dimension
            for (i, &dim_size) in dims.iter().enumerate() {
                if i != self.dim {
                    remaining_dims.push(dim_size);
                }
            }

            // Calculate multi-dimensional index from flat batch index
            for (i, &dim_size) in remaining_dims.iter().enumerate().rev() {
                let coord = batch_idx % dim_size;
                batch_idx /= dim_size;

                // Map back to original dimension index
                let mut orig_dim = i;
                if orig_dim >= self.dim {
                    orig_dim += 1; // Adjust for skipped FFT dimension
                }

                // Calculate contribution to linear index
                // Previous implementation used a for-range loop to compute the product of the
                // remaining dimensions. Clippy flagged it as a needless range loop; using an
                // iterator product both shortens the code and conveys intent clearly.
                let dim_stride: usize = if orig_dim + 1 < dims.len() {
                    dims[orig_dim + 1..].iter().product()
                } else {
                    1
                };
                start += coord * dim_stride;
            }

            start
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
    /// Execute 2D FFT using row-column decomposition
    pub fn fft2_f32(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>> {
        let dims = layout.dims();

        if dims.len() < 2 {
            return Err(
                crate::Error::Msg("2D FFT requires at least 2 dimensions".to_string()).bt(),
            );
        }

        let h = dims[dims.len() - 2];
        let w = dims[dims.len() - 1];

        // For simplicity, assume input is contiguous and handle in simple 2D layout
        // First, apply 1D FFT to each row
        let row_config = CpuFftConfig {
            forward: self.config.forward,
            normalized: false, // Apply normalization only at the end
            real_input: self.config.real_input,
        };

        let mut temp_result = Vec::new();
        let effective_w = if self.config.real_input {
            (w / 2 + 1) * 2
        } else {
            w * 2
        };

        // Process each row
        for row in 0..h {
            let row_start = row * w;
            let row_end = row_start + w;

            if row_end <= input.len() {
                let row_data = &input[row_start..row_end];

                // Create a layout for this row
                let row_layout = Layout::contiguous(vec![w]);
                let row_fft = CpuFft::new(row_config, 0);
                let row_result = row_fft.fft_f32(row_data, &row_layout)?;

                temp_result.extend_from_slice(&row_result);
            } else {
                return Err(crate::Error::Msg(format!(
                    "Row bounds error: trying to access {}..{} in array of length {}",
                    row_start,
                    row_end,
                    input.len()
                ))
                .bt());
            }
        }

        // Now apply 1D FFT to each column
        let col_config = CpuFftConfig {
            forward: self.config.forward,
            normalized: self.config.normalized,
            real_input: false, // Always complex after first FFT
        };

        let mut final_result = vec![0.0; temp_result.len()];
        let temp_w = effective_w / 2; // Number of complex elements per row

        // Process each column
        for col in 0..temp_w {
            let mut col_data = Vec::with_capacity(h * 2);

            // Extract column data (complex interleaved)
            for row in 0..h {
                let idx = row * effective_w + col * 2;
                if idx + 1 < temp_result.len() {
                    col_data.push(temp_result[idx]); // real
                    col_data.push(temp_result[idx + 1]); // imag
                } else {
                    return Err(crate::Error::Msg(format!("Column extraction bounds error: trying to access {}..{} in array of length {}", idx, idx + 2, temp_result.len())).bt());
                }
            }

            // Apply FFT to this column
            let col_layout = Layout::contiguous(vec![h]);
            let col_fft = CpuFft::new(col_config, 0);
            let col_result = col_fft.fft_f32(&col_data, &col_layout)?;

            // Put the result back into the final array
            for (row, chunk) in col_result.chunks(2).enumerate() {
                let idx = row * effective_w + col * 2;
                if idx + 1 < final_result.len() && chunk.len() == 2 {
                    final_result[idx] = chunk[0]; // real
                    final_result[idx + 1] = chunk[1]; // imag
                }
            }
        }

        Ok(final_result)
    }

    /// Transpose a complex matrix stored as interleaved real/imaginary values
    #[allow(dead_code)]
    fn transpose_complex(&self, data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let mut result = vec![0.0; data.len()];

        for r in 0..rows {
            for c in 0..cols {
                let src_idx = (r * cols + c) * 2;
                let dst_idx = (c * rows + r) * 2;

                if src_idx + 1 < data.len() && dst_idx + 1 < result.len() {
                    result[dst_idx] = data[src_idx]; // real part
                    result[dst_idx + 1] = data[src_idx + 1]; // imaginary part
                }
            }
        }

        Ok(result)
    }

    /// Execute FFT along a specific axis
    #[allow(dead_code)]
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
            let factor =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / (window_size - 1) as f32).cos());
            *val *= factor;
        }
    }

    /// Apply Hamming window to input data
    pub fn apply_hamming_window(&self, data: &mut [f32], window_size: usize) {
        for (i, val) in data.iter_mut().enumerate() {
            let n = (i % window_size) as f32;
            let factor =
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * n / (window_size - 1) as f32).cos();
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
