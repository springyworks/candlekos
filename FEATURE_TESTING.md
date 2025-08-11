# Feature-Gated Testing in Candle

This document explains how Candle uses feature gates to manage optional functionality and their corresponding tests.

## FFT Feature Gate

### Overview
The FFT (Fast Fourier Transform) functionality in Candle requires the `fft` feature to be enabled. This feature provides:

- CPU-based FFT operations using RustFFT
- Professional-grade 1D, 2D, and multi-dimensional FFT operations
- Real-to-complex and complex-to-complex transforms
- Windowing functions and spectral analysis tools

### Running FFT Tests

❌ **Without feature** (will show helpful error):
```bash
cargo test --test fft_feature_check
```

✅ **With feature enabled**:
```bash
# Run all FFT tests
cargo test --features fft --test fft_tests

# Run specific FFT test
cargo test --features fft test_cpu_fft_basic

# Run all tests with FFT enabled
cargo test --features fft
```

### Feature Check Test

We provide a dedicated feature check test (`fft_feature_check.rs`) that:

1. **When FFT feature is disabled**: Shows a clear error message with usage instructions
2. **When FFT feature is enabled**: Confirms the feature is working and tests can run

This prevents user confusion when accidentally running tests without required features.

### Test Structure

- `fft_tests.rs`: Main FFT test suite (only compiled with `fft` feature)
- `fft_feature_check.rs`: Feature availability checker (always compiled)

## Scan Operations

### Overview
Scan operations (cumsum, prefix scan, etc.) are **built into the core** and do not require additional features:

- Always available in CPU mode
- Automatically uses CUDA acceleration when available
- No feature gates required

### Running Scan Tests

✅ **Always available**:
```bash
# Run scan tests (no special features needed)
cargo test --test scan_tests

# Scan operations work in any test
cargo test cumsum
```

## Best Practices

### For Users
1. **Check feature requirements**: Look for compilation errors mentioning missing features
2. **Use feature check tests**: Run `cargo test --test <feature>_feature_check` to verify setup
3. **Read error messages**: Our feature gates provide helpful guidance on correct usage

### For Developers
1. **Feature gate optional functionality**: Use `#![cfg(feature = "feature_name")]` for entire test files
2. **Provide feature check tests**: Create helpful tests that guide users when features are missing
3. **Document feature requirements**: Clear documentation of what features enable what functionality

## Summary

| Operation | Feature Required | Test Command |
|-----------|-----------------|--------------|
| FFT       | `fft`           | `cargo test --features fft` |
| Scan      | None (core)     | `cargo test` |
| CUDA      | `cuda`          | `cargo test --features cuda` |

This system ensures that:
- Tests only run when their dependencies are available
- Users get clear guidance when features are missing
- Core functionality remains always accessible
- Optional features enhance capabilities without breaking basic usage
