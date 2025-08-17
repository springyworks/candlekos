# xtask – Candle Workspace Developer Utilities

This crate implements the **xtask pattern** for the Candle workspace. It collects
helper commands used during development & CI, plus a convenience *run-by-path*
launcher for arbitrary Rust source files containing a `main` function.

## Commands

```bash
xtask list            # Show canonical exploration crate feature combos
xtask check           # cargo check over canonical feature sets
xtask check-all       # Broader bounded powerset (size ≤ 3) of features
xtask test            # Build tests (no run) over canonical sets
xtask test-all        # Same as check-all but for tests
xtask lint-workspace  # Run clippy lints across the workspace
xtask comprehensive   # Run comprehensive workspace health check
xtask run-file <path> [cargo flags / -- program args]
xtask --quiet <subcommand>        # Ultra-quiet mode (or set XTASK_QUIET=1)
```

Note: If your terminal prints "notbook" during a long build, that's just our inner compiler goblin making puns. First runs warm the cache; subsequent runs are much faster.

### Workspace Health Commands

The workspace health commands provide comprehensive testing and analysis:

- **`lint-workspace`**: Runs `cargo clippy` with strict linting across all workspace members
- **`comprehensive`**: Executes a full workspace health check including:
  - Workspace compilation check
  - Feature combination testing
  - Test build verification  
  - Clippy linting analysis
  - Documentation build validation
  - Code formatting verification

The comprehensive command provides a detailed summary with pass/fail counts and specific error details for debugging.

Extended mode:

- You can expand coverage with environment variables:
  - `XTASK_COMPREHENSIVE=1` – widens the feature/test matrix and runs additional checks.
  - `XTASK_CORE_FFT=1` – includes candle-core GPU FFT combos (e.g. VkFFT) in the matrix.

Example extended run:

```bash
XTASK_COMPREHENSIVE=1 XTASK_CORE_FFT=1 cargo run -p xtask -- comprehensive
```
- Quiet mode:

You can suppress most warnings and noisy compiler output for third-party code paths (like VkFFT C):

```bash
cargo run -p xtask -- --quiet comprehensive
# or
XTASK_QUIET=1 cargo run -p xtask -- comprehensive
```

Quiet mode sets RUSTFLAGS=-Awarnings and CANDLE_QUIET=1 (recognized by build scripts) to mute benign warnings.


#### How to use ultra‑quiet mode

Quiet for any xtask command:

xtask flag:

```bash
cargo run -p xtask -- --quiet comprehensive
```

or environment:

```bash
XTASK_QUIET=1 cargo run -p xtask -- comprehensive
```


Notes:
- The FFT path may require third-party headers (VkFFT submodule) and CUDA toolchain if you enable GPU-related features.
- First runs will build more artifacts; subsequent runs are much faster.

### run-file details

`run-file` lets you execute a Rust file **just by its path**:

```bash
cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs
```

Behavior:
- Determines the owning workspace crate (deepest manifest ancestor).
- If the file matches a declared `[[bin]]` target, runs it directly.
- If it is under `src/bin/<name>.rs`, leverages Cargo's auto-discovery.
- If it is the crate's `src/main.rs`, runs the package.
- Otherwise copies the file to a temporary `src/bin/__xtask_temp_<stem>.rs`, runs it, then deletes it.
- Automatically enables any `required-features` for the resolved binary target unless you explicitly pass your own `--features ...` flag.
- Splits arguments: cargo flags (like `--release`, `--features`) go before the first standalone `--`; anything after `--` is forwarded to the program.

Examples:

```bash
# Auto features (e.g. enables `cuda` if required)
cargo run -p xtask -- run-file 0aEXPLORATION/tensor_feedback_simple.rs

# Release build + pass a program arg
cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs --release -- --help

# Override features manually (disables auto feature enabling)
cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs --features cuda,fft

# Suppress auto features forcibly (leave feature set empty)
cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs --features=
```

## VS Code Integration (runTerminalCommand extension)

Add entries to your workspace `workspace.json`:

```jsonc
"runTerminalCommand.commands": [
  { "command": "cargo run -p xtask -- run-file {resource}", "name": "Rust: Run current file (auto features)", "group": "Rust" },
  { "command": "cargo run -p xtask -- run-file {resource} --release", "name": "Rust: Run current file (release)", "group": "Rust" },
  { "command": "cargo run -p xtask -- run-file {resource} -- --help", "name": "Rust: Run current file (help)", "group": "Rust" },
  { "command": "cargo run -p xtask -- run-file {resource} --features cuda,fft --release", "name": "Rust: Run current file (cuda+fft release)", "group": "Rust" },
  { "command": "cargo run -p xtask -- run-file {resource} --features= --", "name": "Rust: Run current file (no auto features)", "group": "Rust" },
  { "command": "cargo run -p xtask -- run-file {resource} -- {input:programArgs}", "name": "Rust: Run current file (with args)", "group": "Rust" }
]
```

(Use `{resource}` rather than `${relativeFile}`; this extension uses `{token}` syntax.)

Optional GPU selection variant:

```bash
CANDLE_CUDA_DEVICE=0 cargo run -p xtask -- run-file {resource}
```

## Design Notes
- Keeps feature combination enumeration bounded for CI latency.
- Defers to Cargo for resolution; no custom build scripting.
- Avoids symlinks for portability; uses temporary file copy fallback.

## Future Ideas
- Example (`examples/*.rs`) auto-detection.
- `watch` mode (integrate with `cargo watch`).
- Timing / profiling helpers.

Contributions welcome.

---

Tip: cargo test

This workspace prefers running the xtask comprehensive suite for holistic health checks. If you habitually run `cargo test` at the root, consider instead:

```bash
cargo run -p xtask -- comprehensive
```

Or enable the extended matrix as shown above. A future iteration may add a small test that prints a friendly pointer when invoking `cargo test`.
