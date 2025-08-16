# Feature-Gated Testing in Candle

This document explains how Candle uses feature gates to manage optional functionality and their corresponding tests.

## FFT Feature Gate

### Overview
The FFT (Fast Fourier Transform) functionality in Candle requires the `fft` feature to be enabled. This feature provides:

- CPU-based FFT operations using RustFFT
- 1D, 2D, and multi-dimensional FFT operations
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

## Feature map for examples and runnable binaries

Below is an automated mapping of Rust examples and binary files found in the repository and the feature gates they reference. This is a heuristic scan (searches for `fn main` and `cfg(feature = "...")`) and should be used as a starting point — consult each example's README for exact runtime/assets requirements.

| Path | Type | Feature gates found | README |
|---|---:|---|---|
| `0aEXPLORATION/NOT/fft_overview.rs` | bin |  |  |
| `0aEXPLORATION/code_overview.rs` | bin |  | README.md |
| `0aEXPLORATION/compound_data_examples.rs` | bin |  | README.md |
| `0aEXPLORATION/gpu_fft2_dual_pane.rs` | bin |  | README.md |
| `0aEXPLORATION/gpu_stream_display.rs` | bin |  | README.md |
| `0aEXPLORATION/gpu_tensor_feedback.rs` | bin |  | README.md |
| `0aEXPLORATION/tensor_feedback_simple.rs` | bin |  | README.md |
| `0aEXPLORATION/tensor_feedback_viz.rs` | bin | viz-debug | README.md |
| `0aEXPLORATION/test_custom_types.rs` | bin |  | README.md |
| `0aEXPLORATION/test_higher_dims.rs` | bin |  | README.md |
| `candle-core/build.rs` | bin |  | README.md |
| `candle-core/examples/basics.rs` | example | accelerate, mkl |  |
| `candle-core/examples/cuda_basics.rs` | example | accelerate, mkl |  |
| `candle-core/examples/cuda_sum_benchmark.rs` | example | accelerate, mkl |  |
| `candle-core/examples/metal_basics.rs` | example | accelerate, mkl |  |
| `candle-core/src/lib.rs` | bin | accelerate, cuda, cudnn, metal, mkl |  |
| `candle-examples/build.rs` | bin | cuda | README.md |
| `candle-examples/examples/based/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/beit/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/bert/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/bigcode/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/blip/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/chatglm/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/chinese_clip/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/clip/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/codegeex4-9b/main.rs` | example |  |  |
| `candle-examples/examples/colpali/main.rs` | example |  | README.md |
| `candle-examples/examples/convmixer/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/convnext/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/csm/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/custom-ops/main.rs` | example | cuda, mkl | README.md |
| `candle-examples/examples/debertav2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/deepseekv2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/depth_anything_v2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/dinov2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/dinov2reg4/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/distilbert/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/efficientnet/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/efficientvit/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/encodec/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/eva2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/falcon/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/fastvit/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/flux/main.rs` | example | accelerate, cuda, mkl | README.md |
| `candle-examples/examples/gemma/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/glm4/main.rs` | example |  | README.md |
| `candle-examples/examples/granite/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/gte-qwen/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/helium/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/hiera/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/jina-bert/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/llama/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/llama2-c/main.rs` | example | accelerate, cuda, mkl |  |
| `candle-examples/examples/llama_multiprocess/main.rs` | example | mkl |  |
| `candle-examples/examples/llava/main.rs` | example |  |  |
| `candle-examples/examples/mamba-minimal/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mamba/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/marian-mt/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/metavoice/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mimi/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mistral/main.rs` | example | accelerate, cuda, mkl | README.md |
| `candle-examples/examples/mixtral/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mnist-training/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mobileclip/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mobilenetv4/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/mobileone/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/modernbert/main.rs` | example |  | README.md |
| `candle-examples/examples/moondream/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/musicgen/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/nvembed_v2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/olmo/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/onnx-llm/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/onnx/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/onnx_basics.rs` | example |  |  |
| `candle-examples/examples/orpheus/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/paligemma/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/parler-tts/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/phi/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/pixtral/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/quantized-gemma/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/quantized-phi/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/quantized-qwen2-instruct/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/quantized-qwen3/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/quantized-t5/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/quantized/main.rs` | example | accelerate, cuda, mkl | README.md |
| `candle-examples/examples/qwen/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/recurrent-gemma/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/reinforcement-learning/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/replit-code/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/repvgg/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/resnet/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/rwkv/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/scan.rs` | example |  |  |
| `candle-examples/examples/segformer/main.rs` | example |  | README.md |
| `candle-examples/examples/segment-anything/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/siglip/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/silero-vad/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/snac/main.rs` | example | accelerate, mkl |  |
| `candle-examples/examples/splade/main.rs` | example |  | README.md |
| `candle-examples/examples/stable-diffusion-3/main.rs` | example |  | README.md |
| `candle-examples/examples/stable-diffusion/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/stable-lm/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/starcoder2/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/stella-en-v5/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/t5/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/trocr/main.rs` | example | accelerate, mkl |  |
| `candle-examples/examples/vgg/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/vit/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/voxtral/main.rs` | example | cuda | README.md |
| `candle-examples/examples/whisper-microphone/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/whisper/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/wuerstchen/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/xlm-roberta/main.rs` | example |  |  |
| `candle-examples/examples/yi/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/yolo-v3/main.rs` | example | accelerate, mkl | README.md |
| `candle-examples/examples/yolo-v8/main.rs` | example | accelerate, mkl | README.md |
| `candle-flash-attn/build.rs` | bin |  | README.md |
| `candle-kernels/build.rs` | bin |  | README.md |
| `candle-metal-kernels/examples/metal_benchmarks.rs` | example |  |  |
| `candle-metal-kernels/tmp/affine.rs` | bin |  |  |
| `candle-metal-kernels/tmp/binary.rs` | bin |  |  |
| `candle-metal-kernels/tmp/cast.rs` | bin |  |  |
| `candle-metal-kernels/tmp/unary.rs` | bin |  |  |
| `candle-nn/examples/basic_optimizer.rs` | example | accelerate, mkl |  |
| `candle-nn/examples/cpu_benchmarks.rs` | example | accelerate, mkl |  |
| `candle-nn/src/layer_norm.rs` | bin |  |  |
| `candle-nn/src/linear.rs` | bin |  |  |
| `candle-onnx/build.rs` | bin |  | README.md |
| `candle-pyo3/build.rs` | bin |  | README.md |
| `candle-wasm-examples/bert/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/blip/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/llama2-c/src/bin/app.rs` | bin |  |  |
| `candle-wasm-examples/llama2-c/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/llama2-c/src/bin/worker.rs` | bin |  |  |
| `candle-wasm-examples/moondream/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/phi/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/segment-anything/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/t5/src/bin/m-quantized.rs` | bin |  |  |
| `candle-wasm-examples/t5/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/whisper/src/bin/app.rs` | bin |  |  |
| `candle-wasm-examples/whisper/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/whisper/src/bin/worker.rs` | bin |  |  |
| `candle-wasm-examples/yolo/src/bin/app.rs` | bin |  |  |
| `candle-wasm-examples/yolo/src/bin/m.rs` | bin |  |  |
| `candle-wasm-examples/yolo/src/bin/worker.rs` | bin |  |  |
| `tensor-tools/src/main.rs` | bin |  |  |
