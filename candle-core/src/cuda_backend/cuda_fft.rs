// Fast Fourier Transform (FFT) operations for CUDA tensors
// Uses cuFFT library for high-performance GPU computation

use crate::cuda_backend::{kernel_name, kernels, CudaStorage, CudaStorageSlice, WrapErr};
use crate::{CudaDevice, Result, WithDType};
use candle_kernels as kernels;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::fft::{CudaFft, FftPlan1d, FftPlanManyDesc};
use half::{bf16, f16};
use std::sync::Arc;

/// FFT operation configuration
#[derive(Debug, Clone, Copy)]
pub struct FftConfig {
    pub forward: bool,    // true for forward FFT, false for inverse
    pub normalized: bool, // apply normalization factor
    pub real_input: bool, // real-to-complex FFT (vs complex-to-complex)
}

impl Default for FftConfig {
    fn default() -> Self {
        Self {
            forward: true,
            normalized: true,
            real_input: false,
        }
    }
}

/// CUDA FFT operation implementation
pub struct CudaFft {
    pub config: FftConfig,
    pub dim: usize, // dimension along which to perform FFT
}

impl CudaFft {
    pub fn new(config: FftConfig, dim: usize) -> Self {
        Self { config, dim }
    }

    /// Execute complex-to-complex FFT
    pub fn fft_c2c<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        input: &CudaSlice<cudarc::types::c32>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<cudarc::types::c32>> {
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
        let batch_size: usize = dims.iter().take(self.dim).product::<usize>()
            * dims.iter().skip(self.dim + 1).product::<usize>();

        // Create cuFFT plan
        let plan = dev.fft().plan_fft_1d(n, cudarc::fft::FftType::C2C, batch_size)?;
        
        // Allocate output
        let output = unsafe { dev.alloc::<cudarc::types::c32>(shape.elem_count())? };
        
        // Execute FFT
        if self.config.forward {
            plan.fft(input, &output, cudarc::fft::FftDirection::Forward)?;
        } else {
            plan.fft(input, &output, cudarc::fft::FftDirection::Inverse)?;
        }

        // Apply normalization if requested
        if self.config.normalized {
            let norm_factor = if self.config.forward {
                1.0 / (n as f32).sqrt()
            } else {
                1.0 / (n as f32).sqrt()
            };
            
            // Apply normalization using a custom kernel
            self.apply_normalization(&output, norm_factor, dev, layout)?;
        }

        Ok(output)
    }

    /// Execute real-to-complex FFT
    pub fn fft_r2c(
        &self,
        input: &CudaSlice<f32>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<cudarc::types::c32>> {
        let shape = layout.shape();
        let dims = shape.dims();
        
        let n = dims[self.dim];
        let batch_size: usize = dims.iter().take(self.dim).product::<usize>()
            * dims.iter().skip(self.dim + 1).product::<usize>();

        // For R2C, output size is n/2+1 complex numbers
        let mut output_dims = dims.to_vec();
        output_dims[self.dim] = n / 2 + 1;
        let output_size: usize = output_dims.iter().product();

        // Create cuFFT plan for R2C
        let plan = dev.fft().plan_fft_1d(n, cudarc::fft::FftType::R2C, batch_size)?;
        
        // Allocate output
        let output = unsafe { dev.alloc::<cudarc::types::c32>(output_size)? };
        
        // Execute R2C FFT
        plan.fft(input, &output, cudarc::fft::FftDirection::Forward)?;

        // Apply normalization if requested
        if self.config.normalized {
            let norm_factor = 1.0 / (n as f32).sqrt();
            self.apply_normalization(&output, norm_factor, dev, &layout.with_shape(output_dims))?;
        }

        Ok(output)
    }

    /// Apply normalization factor to complex numbers
    fn apply_normalization(
        &self,
        data: &CudaSlice<cudarc::types::c32>,
        factor: f32,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<()> {
        let func = dev.get_or_load_func("fft_normalize_c32", kernels::FFT)?;
        let cfg = LaunchConfig::for_num_elems(data.len() as u32);
        
        let mut builder = func.builder();
        builder.arg(&(data.len() as u32));
        builder.arg(data);
        builder.arg(&factor);
        
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    pub fn fft(&self) -> &Arc<CudaFft> {
        &self.cudarc().fft()
    }
}
