//! GPU FFT support (provider-based: VkFFT/cuFFT/rocFFT). Enabled via `gpu-fft` (or alias `cuda-fft`).
//! A stub is provided when disabled.

#[cfg(not(feature = "gpu-fft"))]
mod stub {
    use crate::{Result, Layout, bail};
    use crate::cuda_backend::{CudaStorage, CudaStorageSlice, CudaDevice};

    #[derive(Debug, Clone, Copy, Default)]
    pub struct FftConfig {
        pub forward: bool,
        pub normalized: bool,
        pub real_input: bool,
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
            bail!("GPU FFT support is disabled. Rebuild with feature `gpu-fft`. ")
        }

        pub fn fft2_f32(
            &self,
            _input: &CudaStorageSlice,
            _dev: &CudaDevice,
            _layout: &Layout,
        ) -> Result<CudaStorage> {
            bail!("GPU FFT support is disabled. Rebuild with feature `gpu-fft`. ")
        }

        pub fn magnitude(
            &self,
            _complex_input: &CudaStorageSlice,
            _output: &mut CudaStorage,
            _dev: &CudaDevice,
        ) -> Result<()> {
            bail!("GPU FFT support is disabled. Rebuild with feature `gpu-fft`. ")
        }

        pub fn phase(
            &self,
            _complex_input: &CudaStorageSlice,
            _output: &mut CudaStorage,
            _dev: &CudaDevice,
        ) -> Result<()> {
            bail!("GPU FFT support is disabled. Rebuild with feature `gpu-fft`. ")
        }
    }
}

#[cfg(all(feature = "gpu-fft", not(feature = "gpu-fft-vkfft")))]
mod impls;
#[cfg(all(feature = "gpu-fft", feature = "gpu-fft-vkfft"))]
mod vkfft;

#[cfg(not(feature = "gpu-fft"))]
pub use stub::*;
#[cfg(all(feature = "gpu-fft", not(feature = "gpu-fft-vkfft")))]
pub use impls::*;
#[cfg(all(feature = "gpu-fft", feature = "gpu-fft-vkfft"))]
pub use vkfft::*;
