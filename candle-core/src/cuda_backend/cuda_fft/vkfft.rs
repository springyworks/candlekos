//! VkFFT provider (portable across CUDA/HIP/Vulkan/OpenCL/Metal).
//! Minimal scaffold: returns an explicit error until wired to FFI.

use crate::cuda_backend::{CudaStorage, CudaStorageSlice, CudaDevice};
use crate::{Result, Layout, bail};
use crate::backend::BackendDevice; // for .location()
use cudarc::driver::{DevicePtr, DevicePtrMut}; // for device_ptr/device_ptr_mut

#[cfg(feature = "gpu-fft-vkfft-ffi")]
extern "C" {
    fn candle_vkfft_header_sanity() -> ::std::os::raw::c_int;
    fn candle_vkfft_cuda_init_teardown(device_ordinal: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
    fn candle_vkfft_exec_r2c_f32(
        device_ordinal: ::std::os::raw::c_int,
        in_ptr: u64,
        in_offset_bytes: usize,
        out_ptr: u64,
        out_offset_bytes: usize,
        n: usize,
        batch: usize,
        normalized: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}

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
        #[cfg(feature = "gpu-fft-vkfft-ffi")]
        unsafe {
            // Only implement forward real->complex for contiguous f32 as a first step.
            let is_forward = self.config.forward;
            let is_real = self.config.real_input;
            if !(is_forward && is_real) {
                let ver = candle_vkfft_header_sanity();
                bail!("VkFFT FFI enabled (version {ver}), but this path is not implemented yet (only forward real->complex)." );
            }

            // Extract input device pointer and compute batch, n along the last dim of layout.
            let layout = _layout;
            let shape = layout.shape();
            let dims = shape.dims();
            let rank = dims.len();
            let n = dims[rank - 1];
            let batch: usize = if rank > 1 { dims[..rank - 1].iter().product() } else { 1 };

            // Input pointer and offset
            let (in_ptr_u64, guard) = match _input {
                CudaStorageSlice::F32(s) => {
                    let stream = s.stream();
                    let (ptr, g) = s.device_ptr(stream);
                    (ptr, g)
                }
                _ => bail!("VkFFT f32 path received non-f32 storage slice"),
            };
            let start_elem = layout.start_offset();
            let in_offset_bytes = start_elem * core::mem::size_of::<f32>();

            // Allocate output buffer: real->complex packs to (n/2+1)*2 elements along last dim
            let mut out_dims = dims.to_vec();
            out_dims[rank - 1] = (n / 2 + 1) * 2;
            let out_el: usize = out_dims.iter().product();
            let mut out_slice = _dev.alloc::<f32>(out_el)?;
            let stream_out = _dev.cuda_stream();
            let (out_ptr_u64, og) = out_slice.device_ptr_mut(&stream_out);
            let out_offset_bytes = 0usize; // write starting at 0

            // Call wrapper
            let ord = match _dev.location() {
                crate::DeviceLocation::Cuda { gpu_id } => gpu_id,
                _ => 0, // unreachable for CudaDevice
            };
            let res = candle_vkfft_exec_r2c_f32(
                ord as i32,
                in_ptr_u64,
                in_offset_bytes,
                out_ptr_u64,
                out_offset_bytes,
                n,
                batch,
                if self.config.normalized { 1 } else { 0 },
            );
            drop(guard);
            drop(og);
            if res != 0 {
                bail!("VkFFT exec r2c failed with code {res}");
            }

            Ok(CudaStorage { slice: CudaStorageSlice::F32(out_slice), device: _dev.clone() })
        }
        #[cfg(not(feature = "gpu-fft-vkfft-ffi"))]
        {
            bail!("VkFFT provider selected, but FFI is not yet wired. This is a placeholder.")
        }
    }

    pub fn fft2_f32(
        &self,
        _input: &CudaStorageSlice,
        _dev: &CudaDevice,
        _layout: &Layout,
    ) -> Result<CudaStorage> {
        #[cfg(feature = "gpu-fft-vkfft-ffi")]
        unsafe {
            let ver = candle_vkfft_header_sanity();
            bail!("VkFFT FFI enabled (version {ver}), but execution is not implemented yet.");
        }
        #[cfg(not(feature = "gpu-fft-vkfft-ffi"))]
        {
            bail!("VkFFT provider selected, but FFI is not yet wired. This is a placeholder.")
        }
    }

    pub fn magnitude(
        &self,
        _complex_input: &CudaStorageSlice,
        _output: &mut CudaStorage,
        _dev: &CudaDevice,
    ) -> Result<()> {
        #[cfg(feature = "gpu-fft-vkfft-ffi")]
        unsafe {
            let ver = candle_vkfft_header_sanity();
            bail!("VkFFT FFI enabled (version {ver}), but execution is not implemented yet.");
        }
        #[cfg(not(feature = "gpu-fft-vkfft-ffi"))]
        {
            bail!("VkFFT provider selected, but FFI is not yet wired. This is a placeholder.")
        }
    }

    pub fn phase(
        &self,
        _complex_input: &CudaStorageSlice,
        _output: &mut CudaStorage,
        _dev: &CudaDevice,
    ) -> Result<()> {
        #[cfg(feature = "gpu-fft-vkfft-ffi")]
        unsafe {
            let ver = candle_vkfft_header_sanity();
            bail!("VkFFT FFI enabled (version {ver}), but execution is not implemented yet.");
        }
        #[cfg(not(feature = "gpu-fft-vkfft-ffi"))]
        {
            bail!("VkFFT provider selected, but FFI is not yet wired. This is a placeholder.")
        }
    }
}
