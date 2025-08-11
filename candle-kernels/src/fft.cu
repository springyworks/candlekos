// FFT CUDA kernels
// Provides normalization and utility functions for cuFFT operations

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

// Normalization kernel for complex32 data
extern "C" __global__ void fft_normalize_c32(
    const size_t numel,
    cuFloatComplex* data,
    const float factor
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    data[i] = cuCmulf(data[i], make_cuFloatComplex(factor, 0.0f));
}

// Normalization kernel for complex64 data
extern "C" __global__ void fft_normalize_c64(
    const size_t numel,
    cuDoubleComplex* data,
    const double factor
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    data[i] = cuCmul(data[i], make_cuDoubleComplex(factor, 0.0));
}

// Convert real data to complex (interleaved) format
extern "C" __global__ void real_to_complex_f32(
    const size_t numel,
    const float* input,
    cuFloatComplex* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = make_cuFloatComplex(input[i], 0.0f);
}

// Extract real part from complex data
extern "C" __global__ void complex_to_real_f32(
    const size_t numel,
    const cuFloatComplex* input,
    float* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = cuCrealf(input[i]);
}

// Extract imaginary part from complex data
extern "C" __global__ void complex_to_imag_f32(
    const size_t numel,
    const cuFloatComplex* input,
    float* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = cuCimagf(input[i]);
}

// Compute magnitude of complex data
extern "C" __global__ void complex_magnitude_f32(
    const size_t numel,
    const cuFloatComplex* input,
    float* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = cuCabsf(input[i]);
}

// Compute phase of complex data
extern "C" __global__ void complex_phase_f32(
    const size_t numel,
    const cuFloatComplex* input,
    float* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = atan2f(cuCimagf(input[i]), cuCrealf(input[i]));
}

// Apply window function (Hann window) before FFT
extern "C" __global__ void apply_hann_window_f32(
    const size_t numel,
    float* data,
    const size_t window_size
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    // Apply Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
    const float n = i % window_size;
    const float factor = 0.5f * (1.0f - cosf(2.0f * M_PI * n / (window_size - 1)));
    data[i] *= factor;
}

// FFT shift operation (move zero frequency to center)
extern "C" __global__ void fft_shift_c32(
    const size_t numel,
    const cuFloatComplex* input,
    cuFloatComplex* output,
    const size_t fft_size
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    const size_t batch_idx = i / fft_size;
    const size_t freq_idx = i % fft_size;
    const size_t shifted_idx = (freq_idx + fft_size / 2) % fft_size;
    const size_t output_idx = batch_idx * fft_size + shifted_idx;
    
    output[output_idx] = input[i];
}

// Inverse FFT shift operation
extern "C" __global__ void ifft_shift_c32(
    const size_t numel,
    const cuFloatComplex* input,
    cuFloatComplex* output,
    const size_t fft_size
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    const size_t batch_idx = i / fft_size;
    const size_t freq_idx = i % fft_size;
    const size_t shifted_idx = (freq_idx + (fft_size + 1) / 2) % fft_size;
    const size_t output_idx = batch_idx * fft_size + shifted_idx;
    
    output[output_idx] = input[i];
}
