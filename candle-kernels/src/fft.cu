// FFT CUDA kernels
// High-performance implementations for windowing, magnitude/phase extraction, and FFT shift
// Provides normalization and utility functions for cuFFT operations

//! CUDA kernels for FFT support operations including windowing and magnitude computation.
//! Provides optimized GPU kernels for spectral analysis, phase extraction, and frequency domain operations.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>

const int BLOCK_SIZE = 256;

// ============================================================================
// Normalization Kernels (Original + Enhanced)
// ============================================================================

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

// ============================================================================
// Window Function Kernels
// ============================================================================

extern "C" __global__ void apply_hann_window(
    float* data,
    const unsigned int window_size,
    const unsigned int total_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        unsigned int window_pos = idx % window_size;
        float n = static_cast<float>(window_pos);
        float size_minus_1 = static_cast<float>(window_size - 1);
        
        // Hann window: 0.5 * (1 - cos(2π * n / (N-1)))
        float factor = 0.5f * (1.0f - cosf(2.0f * M_PI * n / size_minus_1));
        data[idx] *= factor;
    }
}

extern "C" __global__ void apply_hamming_window(
    float* data,
    const unsigned int window_size,
    const unsigned int total_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        unsigned int window_pos = idx % window_size;
        float n = static_cast<float>(window_pos);
        float size_minus_1 = static_cast<float>(window_size - 1);
        
        // Hamming window: 0.54 - 0.46 * cos(2π * n / (N-1))
        float factor = 0.54f - 0.46f * cosf(2.0f * M_PI * n / size_minus_1);
        data[idx] *= factor;
    }
}

extern "C" __global__ void apply_blackman_window(
    float* data,
    const unsigned int window_size,
    const unsigned int total_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        unsigned int window_pos = idx % window_size;
        float n = static_cast<float>(window_pos);
        float size_minus_1 = static_cast<float>(window_size - 1);
        
        // Blackman window: 0.42 - 0.5*cos(2π*n/(N-1)) + 0.08*cos(4π*n/(N-1))
        float arg = 2.0f * M_PI * n / size_minus_1;
        float factor = 0.42f - 0.5f * cosf(arg) + 0.08f * cosf(2.0f * arg);
        data[idx] *= factor;
    }
}

// ============================================================================
// Complex Number Operations
// ============================================================================

extern "C" __global__ void complex_magnitude_kernel(
    const float* complex_input,
    float* magnitude_output,
    const unsigned int complex_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < complex_count) {
        float real = complex_input[2 * idx];
        float imag = complex_input[2 * idx + 1];
        
        // Magnitude = sqrt(real^2 + imag^2)
        magnitude_output[idx] = sqrtf(real * real + imag * imag);
    }
}

extern "C" __global__ void complex_phase_kernel(
    const float* complex_input,
    float* phase_output,
    const unsigned int complex_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < complex_count) {
        float real = complex_input[2 * idx];
        float imag = complex_input[2 * idx + 1];
        
        // Phase = atan2(imag, real)
        phase_output[idx] = atan2f(imag, real);
    }
}

extern "C" __global__ void complex_power_kernel(
    const float* complex_input,
    float* power_output,
    const unsigned int complex_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < complex_count) {
        float real = complex_input[2 * idx];
        float imag = complex_input[2 * idx + 1];
        
        // Power = real^2 + imag^2
        power_output[idx] = real * real + imag * imag;
    }
}

// ============================================================================
// FFT Shift Operations
// ============================================================================

__device__ unsigned int calculate_shifted_index(
    unsigned int original_idx,
    const unsigned int* shape,
    unsigned int ndim,
    const unsigned int* shift_axes,
    unsigned int num_axes
) {
    // Convert linear index to multi-dimensional coordinates
    unsigned int coords[8]; // Max 8 dimensions
    unsigned int temp_idx = original_idx;
    
    // Calculate coordinates
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_idx % shape[d];
        temp_idx /= shape[d];
    }
    
    // Apply shifts to specified axes
    for (unsigned int i = 0; i < num_axes; i++) {
        unsigned int axis = shift_axes[i];
        if (axis < ndim) {
            unsigned int dim_size = shape[axis];
            unsigned int shift_amount = dim_size / 2;
            coords[axis] = (coords[axis] + shift_amount) % dim_size;
        }
    }
    
    // Convert back to linear index
    unsigned int shifted_idx = 0;
    unsigned int stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        shifted_idx += coords[d] * stride;
        stride *= shape[d];
    }
    
    return shifted_idx;
}

extern "C" __global__ void fftshift_kernel(
    const float* input,
    float* output,
    const unsigned int* shape,
    unsigned int ndim,
    const unsigned int* shift_axes,
    unsigned int num_axes,
    unsigned int total_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        unsigned int shifted_idx = calculate_shifted_index(idx, shape, ndim, shift_axes, num_axes);
        output[shifted_idx] = input[idx];
    }
}

// ============================================================================
// Advanced Normalization
// ============================================================================

extern "C" __global__ void apply_normalization_kernel(
    float* data,
    const float normalization_factor,
    const unsigned int total_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        data[idx] *= normalization_factor;
    }
}

extern "C" __global__ void apply_complex_normalization_kernel(
    float* complex_data,
    const float normalization_factor,
    const unsigned int complex_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < complex_count) {
        complex_data[2 * idx] *= normalization_factor;     // Real part
        complex_data[2 * idx + 1] *= normalization_factor; // Imaginary part
    }
}

// ============================================================================
// Memory Layout Transformations (Legacy Compatible)
// ============================================================================

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

// Compute magnitude of complex data (legacy version using cuFloatComplex)
extern "C" __global__ void complex_magnitude_f32(
    const size_t numel,
    const cuFloatComplex* input,
    float* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = cuCabsf(input[i]);
}

// Compute phase of complex data (legacy version using cuFloatComplex)
extern "C" __global__ void complex_phase_f32(
    const size_t numel,
    const cuFloatComplex* input,
    float* output
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    output[i] = atan2f(cuCimagf(input[i]), cuCrealf(input[i]));
}

// Apply window function (Hann window) before FFT (legacy version)
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

// FFT shift operation (move zero frequency to center) - legacy version
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

// Inverse FFT shift operation - legacy version
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
