# Feature Policy (candle-exploration)

This document defines the feature flags for the `candle-exploration` crate and the invariants that are enforced at compile time.

## Features

| Feature | Purpose | Propagates To | Notes |
|---------|---------|---------------|-------|
| `cuda`  | Enable CUDA GPU backend support. | `candle-core/cuda`, `candle-nn/cuda`, `candle-transformers/cuda` | Safe to combine with `fft`. Required prerequisite for `cudnn`. |
| `cudnn` | Enable cuDNN accelerated kernels. | `candle-core/cudnn`, `candle-nn/cudnn`, `candle-transformers/cudnn` | Implies `cuda` (compile-time checked). |
| `fft`   | Enable FFT operations (CPU RustFFT+RealFFT). | `candle-core/fft` | Standalone CPU ok; combine with `cuda` for GPU tensors (still CPU FFT unless gpu-fft enabled). |
| `gpu-fft` / `cuda-fft` | Enable GPU FFT provider plumbing. | `candle-core/gpu-fft` | Alias: `cuda-fft` -> `gpu-fft`. Requires `cuda`. |
| `gpu-fft-vkfft` | Select VkFFT provider layer. | `candle-core/gpu-fft-vkfft` | Adds portable backend (future multi-API). Requires `gpu-fft`. |
| `gpu-fft-vkfft-ffi` | Enable VkFFT FFI wrapper build. | `candle-core/gpu-fft-vkfft-ffi` | Off by default to avoid extra system deps. |

## Invariants

Implemented in `src/feature_guards.rs`:
- `cudnn` requires `cuda`. Violations produce a `compile_error!` with guidance.

Planned future invariants (illustrative):
- Additional GPU-only experimental flags would require `cuda`.

## Binaries & Required Features

The `Cargo.toml` declares explicit `required-features` for each binary that needs GPU or FFT support. This prevents accidental CPU-only builds from exposing partially working GPU demos.

Examples:
- `gpu_tensor_feedback` requires `cuda`.
- `gpu_fft2_dual_pane` requires `cuda` and `fft`.
- `fft_overview` requires `fft`.

Attempting to build these bins without their required feature set will cause Cargo to skip them (clean failure instead of a runtime panic path).

## Testing Strategy

The `xtask` utility drives a consistent, low‑overhead feature validation workflow:

1. Canonical fast matrix (CI): baseline, `cuda`, `cuda+fft`, `cuda+cudnn`, `fft`.
2. Limited powerset (nightly): subsets (size ≤ 3) to catch unexpected cross‑feature interactions early.
3. Advanced GPU FFT provider stacks (`gpu-fft`, `gpu-fft-vkfft`, `gpu-fft-vkfft-ffi`) are opt‑in via `XTASK_CORE_FFT=1` so normal CI stays fast.

### Commands

Fast canonical matrix:

```bash
cargo run -p xtask -- check
cargo run -p xtask -- test
```

Limited powerset (nightly / deeper):

```bash
cargo run -p xtask -- check-all
cargo run -p xtask -- test-all
```

Enable advanced GPU FFT combos (locally or nightly non‑fatal step):

```bash
XTASK_CORE_FFT=1 cargo run -p xtask -- check
```

Integrates with CI (`.github/workflows/ci.yml` and `nightly-powerset.yml`).

### Feature-Gated Test Conventions

| Operation | Required Feature(s) | Example Command | Notes |
|-----------|---------------------|-----------------|-------|
| FFT (CPU) | `fft` | `cargo test --features fft --test fft_tests` | Uses RustFFT + RealFFT backends |
| FFT feature check | (none) | `cargo test --test fft_feature_check` | Always builds; prints guidance if `fft` missing |
| GPU FFT baseline | `cuda` + `fft` + `gpu-fft` | `cargo test -p candle-core --features "cuda,fft,gpu-fft"` | Provider plumbing (cuFFT / abstraction) |
| GPU FFT (VkFFT) | baseline + `gpu-fft-vkfft` | add feature flag | Alternative provider layer |
| GPU FFT (VkFFT FFI) | prior + `gpu-fft-vkfft-ffi` | add feature flag | Extra FFI wrapper + deps |
| Scan ops | (core) | `cargo test -p candle-core --test scan_tests` | Inclusive/exclusive scan; CUDA accelerates implicitly |
| GPU FFT smoke (real) | `cuda,fft,gpu-fft` | included (gpu_fft_smoke.rs) | Forward+inverse real roundtrip w/ scale detection |
| GPU FFT smoke (c2c) | `cuda,fft,gpu-fft` | included (gpu_fft_smoke_complex.rs) | Complex roundtrip; tolerances & scale handling |

Guidelines:
1. Provide a small always‑compiled feature check test per optional subsystem (done for FFT).
2. Gate large numeric suites behind `#[cfg(feature = ...)]` to keep baseline builds lean.
3. Avoid verbose investigation tests in main suite; convert to concise assertions or ignore/remove.
4. Exercise experimental GPU layers in nightly, non‑blocking steps (continue-on-error) to keep velocity.
5. Document any new feature interactions here and add invariants in `feature_guards.rs`.
6. GPU FFT smoke tests perform scale normalization detection (some providers scale forward/inverse); assertions adapt accordingly.

## SemVer & Stability

-- Adding a new additive, off-by-default feature: MINOR version bump.
-- Tightening an invariant (e.g. new dependency between features): MINOR if it only affects invalid combos; MAJOR if it breaks valid published combinations.
-- Renaming/removing a feature: MAJOR (provide deprecation window when possible).

## Guidelines For Adding New Features

1. Keep them off by default (`default = []`).
2. Propagate downstream feature flags explicitly instead of relying on implicit names.
3. Document the intent, expected performance impact, and any safety caveats.
4. Add compile-time invariants in `feature_guards.rs` for relationships.
5. Update `xtask` matrices and CI workflows.

## Human Verification Checklist (when reviewing a PR adding a feature)
- [ ] Added to `[features]` with empty default.
- [ ] Downstream crates propagate appropriately.
- [ ] Guard(s) or invariants updated.
- [ ] Documentation section added/updated.
- [ ] Tests or examples exercise the new capability.
