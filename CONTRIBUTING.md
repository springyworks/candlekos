# Contributing to Candle (Fork Guidance)

Thanks for your interest in contributing! This fork (springyworks/candlekos) tracks upstream (huggingface/candle) on `main`, and feature work lands on topic branches before opening PRs upstream.

## Branch strategy
- `main` (public): mirrors upstream `huggingface/candle` `main`. Avoid direct changes here.
- Feature branches (public when PR-ready): e.g. `candle-addition-springyworks-16aug2025`. Keep WIP local until ready.
- Local-only branches (private): keep experimental/dev branches unpushed.

## Draft PR workflow (friendly and iterative)
- Prefer opening a Draft PR first to invite early feedback without pressure to merge.
- Keep the title concise and the description welcoming; clearly state it’s exploratory and feedback is appreciated.
- Convert to “Ready for review” when tests/docs are solid and scope is converged.

## Exploration playground
- Use `0aEXPLORATION/` for prototypes, notebooks, and proofs-of-concept (CPU/GPU), including tensor ops like scan and FFT.
- Promote successful experiments into core crates behind feature flags; retain docs/benchmarks explaining tradeoffs.

## Native builds in this fork
Some features (e.g., GPU FFT via Vulkan, or GPU scans) rely on C/C++ components built via CMake/Ninja. See `build/README.md` for details on how these artifacts integrate with Rust crates via `build.rs`, `cc`, and `bindgen`.

## Development setup
- Rust: stable toolchain; run `cargo build` and `cargo test` in crate roots.
- Optional GPU paths: Vulkan SDK for VkFFT/glslang; CUDA for CUDA features.
- Linting: run `cargo clippy --workspace --all-features` and `cargo fmt --all`.

## Tests
- Add unit tests near changes; keep tolerances realistic for numeric code (document rationale).
- Prefer adding minimal repro tests for bug fixes.

## Code of conduct
Follow upstream community standards; be kind and professional.

## License
By contributing, you agree your changes are licensed under the project’s existing licenses.
