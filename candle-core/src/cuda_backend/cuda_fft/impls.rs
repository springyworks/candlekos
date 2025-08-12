use crate::cuda_backend::{CudaStorage, CudaStorageSlice};
use crate::{CudaDevice, Result, Layout, bail};

/// FFT operation configuration
#[derive(Debug, Clone, Copy)]
pub struct FftConfig {
    pub forward: bool,
    pub normalized: bool,
    pub real_input: bool,
}

impl Default for FftConfig {
    fn default() -> Self {
        Self { forward: true, normalized: true, real_input: false }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaFft {
    pub config: FftConfig,
    pub dim: usize,
}

impl CudaFft {
    pub fn new(config: FftConfig, dim: usize) -> Self { Self { config, dim } }

    pub fn fft_f32(
        &self,
        _input: &CudaStorageSlice,
        _dev: &CudaDevice,
        _layout: &Layout,
    ) -> Result<CudaStorage> {
        // TODO: Implement using cuFFT once APIs are available in the cudarc version pinned by the workspace.
        bail!("CUDA FFT implementation requires cuFFT APIs in cudarc. Use CPU FFT (feature `fft`) or enable a compatible setup.")
    }

    pub fn fft2_f32(
        &self,
        _input: &CudaStorageSlice,
        _dev: &CudaDevice,
        _layout: &Layout,
    ) -> Result<CudaStorage> {
        bail!("CUDA FFT implementation requires cuFFT APIs in cudarc. Use CPU FFT (feature `fft`) or enable a compatible setup.")
    }

    pub fn magnitude(
        &self,
        _complex_input: &CudaStorageSlice,
        _output: &mut CudaStorage,
        _dev: &CudaDevice,
    ) -> Result<()> {
        bail!("CUDA FFT implementation requires cuFFT APIs in cudarc. Use CPU FFT (feature `fft`) or enable a compatible setup.")
    }

    pub fn phase(
        &self,
        _complex_input: &CudaStorageSlice,
        _output: &mut CudaStorage,
        _dev: &CudaDevice,
    ) -> Result<()> {
        bail!("CUDA FFT implementation requires cuFFT APIs in cudarc. Use CPU FFT (feature `fft`) or enable a compatible setup.")
    }
}
