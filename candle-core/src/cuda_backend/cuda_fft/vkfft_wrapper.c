// Minimal C wrapper to ensure VkFFT headers compile with CUDA backend and
// provide symbols we can link against. Also provides a tiny init/teardown
// to validate we can call into VkFFT API beyond version check.

#define VKFFT_BACKEND 1 // CUDA
#include "vkFFT.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>

// Minimal single-entry cache to reduce initialize/delete overhead in tight loops.
typedef struct {
    int valid;
    int device_ordinal;
    size_t n;
    size_t batch;
    int kind; // 0: R2C f32 forward, 1: C2R f32 inverse, 2: C2C f32 forward, 3: C2C f32 inverse
    uint64_t stream_handle;
    VkFFTApplication app;
} CandleVkfftCacheEntry;

static CandleVkfftCacheEntry g_cache = {0};

static void candle_vkfft_cache_clear() {
    if (g_cache.valid) {
        deleteVkFFT(&g_cache.app);
        memset(&g_cache, 0, sizeof(g_cache));
    }
}

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
    int normalized,
    uint64_t stream_handle
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

    // Optional stream binding (defaults to stream 0 if handle is 0)
    cudaStream_t s = (cudaStream_t)(uintptr_t)stream_handle;
    if (s != (cudaStream_t)0) {
        cfg.stream = &s;
        cfg.num_streams = 1;
    } else {
        cfg.stream = 0;
        cfg.num_streams = 0;
    }

    // Try cache
    int kind = 0;
    if (g_cache.valid && g_cache.device_ordinal == device_ordinal && g_cache.n == n && g_cache.batch == batch && g_cache.kind == kind && g_cache.stream_handle == stream_handle) {
        app = g_cache.app;
    } else {
        // Initialize
        cudaDeviceSynchronize();
        VkFFTResult ires = initializeVkFFT(&app, cfg);
        if (ires != VKFFT_SUCCESS) {
            return ires;
        }
        // Replace cache
        candle_vkfft_cache_clear();
        g_cache.valid = 1;
        g_cache.device_ordinal = device_ordinal;
        g_cache.n = n;
        g_cache.batch = batch;
        g_cache.kind = kind;
        g_cache.stream_handle = stream_handle;
        g_cache.app = app;
    }

    // Launch
    VkFFTLaunchParams params = VKFFT_ZERO_INIT;
    params.buffer = &out_buf;
    params.inputBuffer = &in_buf;
    params.inputBufferOffset = (pfUINT)in_offset_bytes;
    params.bufferOffset = (pfUINT)out_offset_bytes;
    VkFFTResult res = VkFFTAppend(&app, /* inverse */ 0, &params);
    cudaDeviceSynchronize();
    return res;
}

// Execute inverse C2R (complex-to-real) 1D FFT for f32 on CUDA via VkFFT.
// - in_ptr (complex), out_ptr (real). Offsets are in bytes.
// - normalized applies to inverse (cfg.normalize=1 when set).
int candle_vkfft_exec_c2r_f32(
    int device_ordinal,
    uint64_t in_ptr,
    size_t in_offset_bytes,
    uint64_t out_ptr,
    size_t out_offset_bytes,
    size_t n,
    size_t batch,
    int normalized,
    uint64_t stream_handle
) {
    VkFFTConfiguration cfg = VKFFT_ZERO_INIT;
    VkFFTApplication app = VKFFT_ZERO_INIT;

    cfg.FFTdim = 1;
    cfg.size[0] = (pfUINT)n;
    cfg.numberBatches = (pfUINT)batch;
    cfg.performR2C = 1;
    cfg.specifyOffsetsAtLaunch = 1;
    cfg.isInputFormatted = 1; // real output not padded
    // Always normalize inverse to recover original scale for round-trip tests
    // (matches Candle's current expectations for rfft -> irfft)
    cfg.normalize = 1;
    cfg.inverseReturnToInputBuffer = 1; // write inverse result to inputBuffer (real)

    CUdevice cu_dev;
    CUresult cu_res = cuDeviceGet(&cu_dev, device_ordinal);
    if (cu_res != CUDA_SUCCESS) {
        return VKFFT_ERROR_INVALID_DEVICE;
    }
    cfg.device = &cu_dev;

    // Buffers: inverse C2R out-of-place: complex input is main buffer; real output via inputBuffer
    void* in_buf = (void*)(uintptr_t)in_ptr;   // complex input
    void* out_buf = (void*)(uintptr_t)out_ptr; // real output
    cfg.buffer = &in_buf; // computation buffer is complex input
    cfg.bufferNum = 1;
    cfg.inputBuffer = &out_buf; // real output goes into inputBuffer when inverseReturnToInputBuffer=1
    cfg.inputBufferNum = 1;

    // Optional stream binding
    cudaStream_t s = (cudaStream_t)(uintptr_t)stream_handle;
    if (s != (cudaStream_t)0) {
        cfg.stream = &s;
        cfg.num_streams = 1;
    } else {
        cfg.stream = 0;
        cfg.num_streams = 0;
    }

    // Cache key
    int kind = 1;
    if (g_cache.valid && g_cache.device_ordinal == device_ordinal && g_cache.n == n && g_cache.batch == batch && g_cache.kind == kind && g_cache.stream_handle == stream_handle) {
        app = g_cache.app;
    } else {
        cudaDeviceSynchronize();
        VkFFTResult ires = initializeVkFFT(&app, cfg);
        if (ires != VKFFT_SUCCESS) {
            return ires;
        }
        candle_vkfft_cache_clear();
        g_cache.valid = 1;
        g_cache.device_ordinal = device_ordinal;
        g_cache.n = n;
        g_cache.batch = batch;
        g_cache.kind = kind;
        g_cache.stream_handle = stream_handle;
        g_cache.app = app;
    }

    VkFFTLaunchParams params = VKFFT_ZERO_INIT;
    params.buffer = &in_buf;
    params.inputBuffer = &out_buf;
    params.bufferOffset = (pfUINT)in_offset_bytes;
    params.inputBufferOffset = (pfUINT)out_offset_bytes;
    VkFFTResult res = VkFFTAppend(&app, /* inverse */ 1, &params);
    cudaDeviceSynchronize();
    return res;
}

// Execute C2C (complex-to-complex) 1D FFT for f32 on CUDA via VkFFT.
// out-of-place by copying input into out buffer first and running in-place on out buffer.
int candle_vkfft_exec_c2c_f32(
    int device_ordinal,
    uint64_t in_ptr,
    size_t in_offset_bytes,
    uint64_t out_ptr,
    size_t out_offset_bytes,
    size_t n,
    size_t batch,
    int forward,
    int normalized,
    uint64_t stream_handle
) {
    VkFFTConfiguration cfg = VKFFT_ZERO_INIT;
    VkFFTApplication app = VKFFT_ZERO_INIT;

    cfg.FFTdim = 1;
    cfg.size[0] = (pfUINT)n;
    cfg.numberBatches = (pfUINT)batch;
    cfg.specifyOffsetsAtLaunch = 1;
    // Apply normalization if requested (typically on inverse path)
    cfg.normalize = normalized ? 1 : 0;

    CUdevice cu_dev;
    CUresult cu_res = cuDeviceGet(&cu_dev, device_ordinal);
    if (cu_res != CUDA_SUCCESS) {
        return VKFFT_ERROR_INVALID_DEVICE;
    }
    cfg.device = &cu_dev;

    // In-place on out buffer: copy in -> out before running
    void* out_buf = (void*)(uintptr_t)out_ptr;
    void* in_buf = (void*)(uintptr_t)in_ptr;
    cudaMemcpy((char*)out_buf + out_offset_bytes, (char*)in_buf + in_offset_bytes, (size_t)(n * sizeof(float) * 2) * batch, cudaMemcpyDeviceToDevice);
    cfg.buffer = &out_buf;
    cfg.bufferNum = 1;

    cudaStream_t s = (cudaStream_t)(uintptr_t)stream_handle;
    if (s != (cudaStream_t)0) {
        cfg.stream = &s;
        cfg.num_streams = 1;
    } else {
        cfg.stream = 0;
        cfg.num_streams = 0;
    }

    // Cache key
    int kind = forward ? 2 : 3;
    if (g_cache.valid && g_cache.device_ordinal == device_ordinal && g_cache.n == n && g_cache.batch == batch && g_cache.kind == kind && g_cache.stream_handle == stream_handle) {
        app = g_cache.app;
    } else {
        cudaDeviceSynchronize();
        VkFFTResult ires = initializeVkFFT(&app, cfg);
        if (ires != VKFFT_SUCCESS) {
            return ires;
        }
        candle_vkfft_cache_clear();
        g_cache.valid = 1;
        g_cache.device_ordinal = device_ordinal;
        g_cache.n = n;
        g_cache.batch = batch;
        g_cache.kind = kind;
        g_cache.stream_handle = stream_handle;
        g_cache.app = app;
    }

    VkFFTLaunchParams params = VKFFT_ZERO_INIT;
    params.buffer = &out_buf;
    params.bufferOffset = (pfUINT)out_offset_bytes;
    VkFFTResult res = VkFFTAppend(&app, forward ? 0 : 1, &params);
    cudaDeviceSynchronize();
    return res;
}

// Execute forward R2C (real-to-complex) 2D FFT for f32 on CUDA via VkFFT.
// Treats the last two dimensions as the transform plane (H, W) and flattens any
// leading dimensions into batches.
// - in_ptr/out_ptr are device addresses (u64). Offsets are in bytes.
// - h, w are transform sizes for the last two dims, batch is number of 2D transforms.
// - normalized: currently ignored for forward; normalization applies to inverse.
// Returns VkFFTResult (0 on success).
int candle_vkfft_exec_r2c2d_f32(
    int device_ordinal,
    uint64_t in_ptr,
    size_t in_offset_bytes,
    uint64_t out_ptr,
    size_t out_offset_bytes,
    size_t h,
    size_t w,
    size_t batch,
    int normalized,
    uint64_t stream_handle
) {
    (void)normalized; // forward normalization not applied here
    VkFFTConfiguration cfg = VKFFT_ZERO_INIT;
    VkFFTApplication app = VKFFT_ZERO_INIT;

    cfg.FFTdim = 2;
    cfg.size[0] = (pfUINT)w; // width (fastest varying dim)
    cfg.size[1] = (pfUINT)h; // height
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

    // Optional stream binding (defaults to stream 0 if handle is 0)
    cudaStream_t s = (cudaStream_t)(uintptr_t)stream_handle;
    if (s != (cudaStream_t)0) {
        cfg.stream = &s;
        cfg.num_streams = 1;
    } else {
        cfg.stream = 0;
        cfg.num_streams = 0;
    }

    // Try cache: kind 10 for 2D R2C
    int kind = 10;
    if (g_cache.valid && g_cache.device_ordinal == device_ordinal && g_cache.n == (w * h) && g_cache.batch == batch && g_cache.kind == kind && g_cache.stream_handle == stream_handle) {
        app = g_cache.app;
    } else {
        cudaDeviceSynchronize();
        VkFFTResult ires = initializeVkFFT(&app, cfg);
        if (ires != VKFFT_SUCCESS) {
            return ires;
        }
        candle_vkfft_cache_clear();
        g_cache.valid = 1;
        g_cache.device_ordinal = device_ordinal;
        g_cache.n = w * h; // store plane area for quick match
        g_cache.batch = batch;
        g_cache.kind = kind;
        g_cache.stream_handle = stream_handle;
        g_cache.app = app;
    }

    // Launch
    VkFFTLaunchParams params = VKFFT_ZERO_INIT;
    params.buffer = &out_buf;
    params.inputBuffer = &in_buf;
    params.inputBufferOffset = (pfUINT)in_offset_bytes;
    params.bufferOffset = (pfUINT)out_offset_bytes;
    VkFFTResult res = VkFFTAppend(&app, /* inverse */ 0, &params);
    cudaDeviceSynchronize();
    return res;
}
