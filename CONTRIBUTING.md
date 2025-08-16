# Contributing to Candle (Fork Guidance)

Thanks for your interest in contributing! This fork (springyworks/candlekos) tracks upstream (huggingface/candle) on `main`, and feature work lands on topic branches before opening PRs upstream.

## Branch strategy
- `main` (public): mirrors upstream `huggingface/candle` `main`. Avoid direct changes here.
- Feature branches (public when PR-ready): e.g. `candle-addition-springyworks-16aug2025`. Keep WIP local until ready.
- Local-only branches (private): keep experimental/dev branches unpushed.

## Opening a friendly PR upstream
When proposing changes to upstream:
1. Rebase/merge upstream `main` so your branch is up-to-date.
2. Keep commits tidy and scoped; include rationale in commit messages.
3. Open a PR from `springyworks:candle-...` to `huggingface:main` with a concise title and polite summary:
   - Problem statement and motivation.
   - What changed (high level), and why this approach.
   - Any feature flags, defaults, or compatibility notes.
   - Benchmarks/test evidence where relevant.
4. Be respectful and responsive to review feedback. Aim for constructive dialogue.

## Native builds in this fork
Some features (e.g., GPU FFT via Vulkan) rely on C/C++ components built via CMake/Ninja. See `build/README.md` for details on VkFFT and glslang targets and how Rust crates link to them via `build.rs`, `cc`, and `bindgen`.

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
