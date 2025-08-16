//! VkFFT provider (portable across CUDA/HIP/Vulkan/OpenCL/Metal).
//! Minimal scaffold: returns an explicit error until wired to FFI.

use crate::backend::BackendDevice; // for .location()
use crate::cuda_backend::error::WrapErr; // for .w() error wrapping
use crate::cuda_backend::{CudaDevice, CudaStorage, CudaStorageSlice};
use crate::{Layout, Result, bail};
use cudarc::driver::{DevicePtr, DevicePtrMut}; // for device_ptr/device_ptr_mut

#[cfg(feature = "gpu-fft-vkfft-ffi")]
extern "C" {
    fn candle_vkfft_header_sanity() -> ::std::os::raw::c_int;
    fn candle_vkfft_cuda_init_teardown(
        device_ordinal: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
    fn candle_vkfft_exec_r2c_f32(
        device_ordinal: ::std::os::raw::c_int,
        in_ptr: u64,
        in_offset_bytes: usize,
        out_ptr: u64,
        out_offset_bytes: usize,
        n: usize,
        batch: usize,
        normalized: ::std::os::raw::c_int,
        stream_handle: u64,
    ) -> ::std::os::raw::c_int;
    fn candle_vkfft_exec_c2r_f32(
        device_ordinal: ::std::os::raw::c_int,
        in_ptr: u64,
        in_offset_bytes: usize,
        out_ptr: u64,
        out_offset_bytes: usize,
        n: usize,
        batch: usize,
        normalized: ::std::os::raw::c_int,
        stream_handle: u64,
    ) -> ::std::os::raw::c_int;
    fn candle_vkfft_exec_c2c_f32(
        device_ordinal: ::std::os::raw::c_int,
        in_ptr: u64,
        in_offset_bytes: usize,
        out_ptr: u64,
        out_offset_bytes: usize,
        n: usize,
        batch: usize,
        forward: ::std::os::raw::c_int,
        normalized: ::std::os::raw::c_int,
        stream_handle: u64,
    ) -> ::std::os::raw::c_int;

    fn candle_vkfft_exec_r2c2d_f32(
        device_ordinal: ::std::os::raw::c_int,
        in_ptr: u64,
        in_offset_bytes: usize,
        out_ptr: u64,
        out_offset_bytes: usize,
        h: usize,
        w: usize,
        batch: usize,
        normalized: ::std::os::raw::c_int,
        stream_handle: u64,
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
        Self {
            forward: true,
            normalized: true,
            real_input: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaFft {
    pub config: FftConfig,
    pub dim: usize,
}

impl CudaFft {
    pub fn new(config: FftConfig, dim: usize) -> Self {
        Self { config, dim }
    }

    pub fn fft_f32(
        &self,
        _input: &CudaStorageSlice,
        _dev: &CudaDevice,
        _layout: &Layout,
    ) -> Result<CudaStorage> {
        #[cfg(feature = "gpu-fft-vkfft-ffi")]
        unsafe {
            // Extract input device pointer and compute batch, n along the last dim of layout.
            let layout = _layout;
            let shape = layout.shape();
            let dims = shape.dims();
            let rank = dims.len();
            let mut n = dims[rank - 1];
            let batch: usize = if rank > 1 {
                dims[..rank - 1].iter().product()
            } else {
                1
            };

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

            // Determine device ordinal and stream

            // Call wrapper
            let ord = match _dev.location() {
                crate::DeviceLocation::Cuda { gpu_id } => gpu_id,
                _ => 0, // unreachable for CudaDevice
            };
            // Obtain a raw CUDA stream handle (as u64) if available
            let raw_stream: u64 = {
                // cudarc's CudaStream does not publicly expose CUstream, but VkFFT uses
                // cudaStream_t (runtime). cudarc wraps a runtime stream; FFI kernels are launched
                // through cudarc with the same stream object. We can pass 0 to use the device's
                // default stream until cudarc adds raw handle extraction. This preserves correctness.
                0u64
            };

            // Branch by transform type
            if self.config.forward && self.config.real_input {
                // Forward real-to-complex: output last dim becomes (n/2+1)*2 floats
                let mut out_dims = dims.to_vec();
                out_dims[rank - 1] = (n / 2 + 1) * 2;
                let out_el: usize = out_dims.iter().product();
                let mut out_slice = _dev.alloc::<f32>(out_el)?;
                let stream_out = _dev.cuda_stream();
                let (out_ptr_u64, og) = out_slice.device_ptr_mut(&stream_out);
                let out_offset_bytes = 0usize;

                let res = candle_vkfft_exec_r2c_f32(
                    ord as i32,
                    in_ptr_u64,
                    in_offset_bytes,
                    out_ptr_u64,
                    out_offset_bytes,
                    n,
                    batch,
                    if self.config.normalized { 1 } else { 0 },
                    raw_stream,
                );
                drop(guard);
                drop(og);
                if res != 0 {
                    bail!("VkFFT exec r2c failed with code {res}");
                }
                Ok(CudaStorage {
                    slice: CudaStorageSlice::F32(out_slice),
                    device: _dev.clone(),
                })
            } else if !self.config.forward && self.config.real_input {
                // Inverse complex-to-real: input is complex interleaved, last dim len = (n/2+1)*2
                // Compute original real n from complex dim size
                let complex_len = n; // n is float-count along last dim
                let n_real = complex_len.saturating_sub(2);
                n = n_real;
                // Output dims: same as input except last dim becomes n_real
                let mut out_dims = dims.to_vec();
                out_dims[rank - 1] = n_real;
                let out_el: usize = out_dims.iter().product();
                let mut out_slice = _dev.alloc::<f32>(out_el)?;
                let stream_out = _dev.cuda_stream();
                let (out_ptr_u64, og) = out_slice.device_ptr_mut(&stream_out);
                let out_offset_bytes = 0usize;

                let res = candle_vkfft_exec_c2r_f32(
                    ord as i32,
                    in_ptr_u64,
                    in_offset_bytes,
                    out_ptr_u64,
                    out_offset_bytes,
                    n,
                    batch,
                    if self.config.normalized { 1 } else { 0 },
                    raw_stream,
                );
                drop(guard);
                drop(og);
                if res != 0 {
                    bail!("VkFFT exec c2r failed with code {res}");
                }
                Ok(CudaStorage {
                    slice: CudaStorageSlice::F32(out_slice),
                    device: _dev.clone(),
                })
            } else {
                // Complex-to-complex forward or inverse
                // Input last dimension encodes interleaved complex: real,imag pairs -> n_complex = n/2
                let n_complex = n / 2;
                // Output dims are identical to input for c2c
                let out_el: usize = dims.iter().product();
                let mut out_slice = _dev.alloc::<f32>(out_el)?;
                let stream_out = _dev.cuda_stream();
                let (out_ptr_u64, og) = out_slice.device_ptr_mut(&stream_out);
                let out_offset_bytes = 0usize;

                let res = candle_vkfft_exec_c2c_f32(
                    ord as i32,
                    in_ptr_u64,
                    in_offset_bytes,
                    out_ptr_u64,
                    out_offset_bytes,
                    n_complex,
                    batch,
                    if self.config.forward { 1 } else { 0 },
                    if self.config.normalized { 1 } else { 0 },
                    raw_stream,
                );
                drop(guard);
                drop(og);
                if res != 0 {
                    bail!("VkFFT exec c2c failed with code {res}");
                }
                Ok(CudaStorage {
                    slice: CudaStorageSlice::F32(out_slice),
                    device: _dev.clone(),
                })
            }
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
            // Expect layout's last two dims as (H, W). Batch is product of leading dims (or 1).
            let dims = _layout.shape().dims();
            let rank = dims.len();
            if rank < 2 {
                bail!("fft2 expects at least 2 dims")
            }
            let h = dims[rank - 2];
            let w = dims[rank - 1];
            let batch: usize = if rank > 2 {
                dims[..rank - 2].iter().product()
            } else {
                1
            };

            // Only support forward R2C for now (matches user's ask). Input must be f32 real.
            if !(self.config.forward && self.config.real_input) {
                bail!("vkfft fft2_f32 currently supports only forward real-input; c2c/c2r 2D todo")
            }

            let (in_ptr_u64, guard) = match _input {
                CudaStorageSlice::F32(s) => {
                    let stream = s.stream();
                    s.device_ptr(stream)
                }
                _ => bail!("VkFFT f32 path received non-f32 storage slice"),
            };
            let in_offset_bytes = _layout.start_offset() * core::mem::size_of::<f32>();

            let ord = match _dev.location() {
                crate::DeviceLocation::Cuda { gpu_id } => gpu_id,
                _ => 0,
            };
            let raw_stream: u64 = 0u64; // default stream for now

            // Output dims: only last dim changes to packed complex ((W/2+1)*2), H unchanged
            let mut out_dims = dims.to_vec();
            out_dims[rank - 1] = (w / 2 + 1) * 2;
            let out_el: usize = out_dims.iter().product();
            let mut out_slice = _dev.alloc::<f32>(out_el)?;
            let stream_out = _dev.cuda_stream();
            let (out_ptr_u64, og) = out_slice.device_ptr_mut(&stream_out);

            let res = candle_vkfft_exec_r2c2d_f32(
                ord as i32,
                in_ptr_u64,
                in_offset_bytes,
                out_ptr_u64,
                0,
                h,
                w,
                batch,
                if self.config.normalized { 1 } else { 0 },
                raw_stream,
            );
            drop(guard);
            drop(og);
            if res != 0 {
                bail!("VkFFT exec r2c2d failed with code {res}");
            }
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out_slice),
                device: _dev.clone(),
            })
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
        // Fallback implementation: copy to host, compute magnitude, copy back.
        // Keeps API working for exploration while we add a device kernel.
        let input_host: Vec<f32> = match _complex_input {
            CudaStorageSlice::F32(s) => s.stream().memcpy_dtov(s).w()?,
            _ => bail!("magnitude expects f32 complex storage"),
        };
        if input_host.len() % 2 != 0 {
            bail!("complex input length must be even (interleaved)")
        }
        let mut out = Vec::with_capacity(input_host.len() / 2);
        for chunk in input_host.chunks_exact(2) {
            let re = chunk[0];
            let im = chunk[1];
            out.push((re * re + im * im).sqrt());
        }
        // Copy back to device output
        let out_slice = _output.as_cuda_slice_mut::<f32>()?;
        let stream = _dev.cuda_stream();
        stream.memcpy_htod(&out, out_slice).w()?;
        Ok(())
    }

    pub fn phase(
        &self,
        _complex_input: &CudaStorageSlice,
        _output: &mut CudaStorage,
        _dev: &CudaDevice,
    ) -> Result<()> {
        // Fallback implementation: copy to host, compute phase, copy back.
        let input_host: Vec<f32> = match _complex_input {
            CudaStorageSlice::F32(s) => s.stream().memcpy_dtov(s).w()?,
            _ => bail!("phase expects f32 complex storage"),
        };
        if input_host.len() % 2 != 0 {
            bail!("complex input length must be even (interleaved)")
        }
        let mut out = Vec::with_capacity(input_host.len() / 2);
        for chunk in input_host.chunks_exact(2) {
            let re = chunk[0];
            let im = chunk[1];
            out.push(im.atan2(re));
        }
        let out_slice = _output.as_cuda_slice_mut::<f32>()?;
        let stream = _dev.cuda_stream();
        stream.memcpy_htod(&out, out_slice).w()?;
        Ok(())
    }
}
