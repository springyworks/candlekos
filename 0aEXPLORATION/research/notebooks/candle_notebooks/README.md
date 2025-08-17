# Candle Rust Notebooks

Rust-native Jupyter notebooks powered by the `evcxr` kernel.

- First runs may be slow as dependencies compile. Subsequent runs are cached.
- Our xtask command can execute notebooks in a single shared evcxr session:
  - `cargo run -p xtask -- test-notebooks`
- Pure Rust execution, no Python shells required.

Heads-up: You may occasionally see a cheeky "notbook" instead of "notebook" in terminal logs during long compiles. Consider it a friendly pun while the build finishes.
