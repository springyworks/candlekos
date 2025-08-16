# xtask – Candle Workspace Developer Utilities

This crate implements the **xtask pattern** for the Candle workspace. It collects
helper commands used during development & CI, plus a convenience *run-by-path*
launcher for arbitrary Rust source files containing a `main` function.

## Commands

```
xtask list            # Show canonical exploration crate feature combos
xtask check           # cargo check over canonical feature sets
xtask check-all       # Broader bounded powerset (size ≤ 3) of features
xtask test            # Build tests (no run) over canonical sets
xtask test-all        # Same as check-all but for tests
xtask lint-workspace  # Run clippy lints across the workspace
xtask comprehensive   # Run comprehensive workspace health check
xtask run-file <path> [cargo flags / -- program args]
```

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

### run-file details

`run-file` lets you execute a Rust file **just by its path**:

```
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
```
# Auto features (e.g. enables `cuda` if required)
cargo run -p xtask -- run-file 0aEXPLORATION/tensor_feedback_simple.rs

# Release build + pass a program arg
you@host$ cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs --release -- --help

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
```
{ "command": "CANDLE_CUDA_DEVICE=0 cargo run -p xtask -- run-file {resource}", "name": "Rust: Run current file (GPU 0)", "group": "Rust" }
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
