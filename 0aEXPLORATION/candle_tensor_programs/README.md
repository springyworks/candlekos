# candle_tensor_programs

Exploration crate for applying tiny expression programs to Candle tensors.

- Feature `lisp` (default): parse a small subset of s-expressions with `lexpr` and
  evaluate them element-wise over a tensor. Supports +, -, *, /, pow, sin, cos, tanh,
  exp, log, abs, sqrt, sqr. Variables: `x` (the current element), optional
  constants (pi, e).
- Feature `rust-eval` (planned): hook for future evcxr-based experiments.

This is research code; API is unstable and may change at any time.
