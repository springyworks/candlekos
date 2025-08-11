// Parallel prefix-sum (scan) kernels (Blelloch) per line along a chosen dimension.
// Based on GPU Gems 3, Chapter 39 (work-efficient scan). This implementation
// processes each independent scan line (all elements sharing all coords except
// along `dim`) in a separate block. Constraints (initial version):
//  - The scanned dimension length must be <= BLOCK_DIM (currently 1024).
//  - Tensor is treated via dims/strides metadata; supports non-contiguous
//    layouts provided strides are valid.
//  - Each block performs an in-shared-memory scan of one line using Blelloch
//    upsweep/downsweep for exclusive scan; inclusive derived from exclusive.
//  - For larger dimensions or when more performance is required (e.g. >1024),
//    a hierarchical multi-block algorithm would be needed (not yet provided).

#include <stdint.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#ifndef SCAN_BLOCK_DIM
#define SCAN_BLOCK_DIM 1024
#endif

template <typename T, bool INCLUSIVE>
__device__ void scan_line_kernel(
    size_t numel,
    size_t storage_len,
    size_t ndims,
    const size_t* __restrict__ dims_and_strides,
    const T* __restrict__ inp,
    T* __restrict__ out,
    size_t dim,
    unsigned long long* __restrict__ debug_ptr)
{
    const size_t* dims = dims_and_strides;
    const size_t* strides = dims_and_strides + ndims;
    size_t dim_len = dims[dim];
    size_t tid = threadIdx.x;
    size_t line_idx = blockIdx.x;
    size_t num_lines = dim_len ? (numel / dim_len) : 0;

    if (line_idx >= num_lines) return;

    size_t remaining = line_idx;
    size_t base_offset = 0;
    for (size_t d = 0; d < ndims; ++d) {
        if (d == dim) continue;
        size_t coord = remaining % dims[d];
        remaining /= dims[d];
        base_offset += coord * strides[d];
    }

    using BlockScanT = cub::BlockScan<T, SCAN_BLOCK_DIM>;
    __shared__ typename BlockScanT::TempStorage temp_storage;
    T carry = T(0);
    T segment_sum_storage; // a place to store the sum of the segment

    for (size_t segment_start = 0; segment_start < dim_len; segment_start += SCAN_BLOCK_DIM) {
        size_t idx = segment_start + tid;
        bool in_range = idx < dim_len;
        T x = T(0);
        if (in_range) {
            size_t addr = base_offset + idx * strides[dim];
            if (addr < storage_len) {
                x = inp[addr];
            } else if (debug_ptr) {
                atomicCAS(debug_ptr, 0ULL, (unsigned long long)addr);
            }
        }
        __syncthreads();

        T scanned_val;
        
        // Perform an inclusive scan to get both the per-thread result and the segment sum.
        BlockScanT(temp_storage).InclusiveScan(x, scanned_val, cub::Sum());
        __syncthreads();

        // The last thread of the valid data holds the sum of the segment.
        size_t last_idx_in_segment = min((size_t)SCAN_BLOCK_DIM, dim_len - segment_start) - 1;
        if (tid == last_idx_in_segment) {
            segment_sum_storage = scanned_val;
        }
        __syncthreads();

        if (in_range) {
            T final_val = INCLUSIVE ? scanned_val : (scanned_val - x);
            final_val += carry;
            size_t addr = base_offset + idx * strides[dim];
            if (addr < storage_len) {
                out[addr] = final_val;
            }
        }
        
        carry += segment_sum_storage;
        __syncthreads();
    }
}

// Host-visible wrappers
extern "C" __global__ void scan_inclusive_f32(
    size_t numel, size_t storage_len, size_t ndims,
    const size_t* dims_and_strides,
    const float* inp, float* out, size_t dim, unsigned long long* dbg)
{
  scan_line_kernel<float, true>(numel, storage_len, ndims,
                                dims_and_strides, inp, out, dim, dbg);
}

extern "C" __global__ void scan_exclusive_f32(
    size_t numel, size_t storage_len, size_t ndims,
    const size_t* dims_and_strides,
    const float* inp, float* out, size_t dim, unsigned long long* dbg)
{
  scan_line_kernel<float, false>(numel, storage_len, ndims,
                                 dims_and_strides, inp, out, dim, dbg);
}

extern "C" __global__ void scan_inclusive_f64(
    size_t numel, size_t storage_len, size_t ndims,
    const size_t* dims_and_strides,
    const double* inp, double* out, size_t dim, unsigned long long* dbg)
{
  scan_line_kernel<double, true>(numel, storage_len, ndims,
                                 dims_and_strides, inp, out, dim, dbg);
}

extern "C" __global__ void scan_exclusive_f64(
    size_t numel, size_t storage_len, size_t ndims,
    const size_t* dims_and_strides,
    const double* inp, double* out, size_t dim, unsigned long long* dbg)
{
  scan_line_kernel<double, false>(numel, storage_len, ndims,
                                  dims_and_strides, inp, out, dim, dbg);
}
