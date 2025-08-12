// Minimal C wrapper to ensure VkFFT headers compile with CUDA backend and
// provide symbols we can link against. Also provides a tiny init/teardown
// to validate we can call into VkFFT API beyond version check.

#define VKFFT_BACKEND 1 // CUDA
#include "vkFFT.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

int candle_vkfft_header_sanity() {
    return VkFFTGetVersion();
}

// Performs a tiny initialize/delete cycle on VkFFT for CUDA backend to validate
// linking and runtime. Uses a dummy 1D size=4 plan, single buffer. Returns VkFFTResult.
int candle_vkfft_cuda_init_teardown(int device_ordinal) {
    VkFFTConfiguration cfg = VKFFT_ZERO_INIT;
    VkFFTApplication app = VKFFT_ZERO_INIT;

    // Configure a trivial 1D plan
    cfg.FFTdim = 1;
    cfg.size[0] = 4;

    // Acquire CUDA device handle from ordinal.
    CUdevice cu_dev;
    CUresult cu_res = cuDeviceGet(&cu_dev, device_ordinal);
    if (cu_res != CUDA_SUCCESS) {
        return VKFFT_ERROR_INVALID_DEVICE;
    }
    cfg.device = &cu_dev;
    // No explicit stream; use default stream (num_streams = 0)
    cfg.stream = 0;
    cfg.num_streams = 0;

    // Initialize and immediately delete (no append)
    cudaDeviceSynchronize();
    VkFFTResult res = initializeVkFFT(&app, cfg);
    if (res != VKFFT_SUCCESS) {
        return res;
    }
    deleteVkFFT(&app);
    cudaDeviceSynchronize();
    return VKFFT_SUCCESS;
}

// Execute forward R2C (real-to-complex) 1D FFT for f32 on CUDA via VkFFT.
// - in_ptr/out_ptr are device addresses (u64). Offsets are in bytes.
// - n is FFT length, batch is number of independent transforms.
// - normalized: currently ignored for forward; normalization applies to inverse.
// Returns VkFFTResult (0 on success).
int candle_vkfft_exec_r2c_f32(
    int device_ordinal,
    uint64_t in_ptr,
    size_t in_offset_bytes,
    uint64_t out_ptr,
    size_t out_offset_bytes,
    size_t n,
    size_t batch,
    int normalized
) {
    (void)normalized; // forward normalization not applied here
    VkFFTConfiguration cfg = VKFFT_ZERO_INIT;
    VkFFTApplication app = VKFFT_ZERO_INIT;

    cfg.FFTdim = 1;
    cfg.size[0] = (pfUINT)n;
    cfg.numberBatches = (pfUINT)batch;
    cfg.performR2C = 1;
    cfg.specifyOffsetsAtLaunch = 1;
    // Indicate that real input is not R2C-padded: provide inputBuffer explicitly.
    cfg.isInputFormatted = 1;

    CUdevice cu_dev;
    CUresult cu_res = cuDeviceGet(&cu_dev, device_ordinal);
    if (cu_res != CUDA_SUCCESS) {
        return VKFFT_ERROR_INVALID_DEVICE;
    }
    cfg.device = &cu_dev;

    // Buffers: out-of-place R2C
    void* in_buf = (void*)(uintptr_t)in_ptr;   // real input
    void* out_buf = (void*)(uintptr_t)out_ptr; // complex output
    cfg.buffer = &out_buf; // main computation/output buffer is complex
    cfg.bufferNum = 1;
    cfg.inputBuffer = &in_buf; // separate real input buffer
    cfg.inputBufferNum = 1;

    // Initialize
    cudaDeviceSynchronize();
    VkFFTResult res = initializeVkFFT(&app, cfg);
    if (res != VKFFT_SUCCESS) {
        return res;
    }

    // Launch
    VkFFTLaunchParams params = VKFFT_ZERO_INIT;
    params.buffer = &out_buf;
    params.inputBuffer = &in_buf;
    params.inputBufferOffset = (pfUINT)in_offset_bytes;
    params.bufferOffset = (pfUINT)out_offset_bytes;
    res = VkFFTAppend(&app, /* inverse */ 0, &params);
    deleteVkFFT(&app);
    cudaDeviceSynchronize();
    return res;
}
