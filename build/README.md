# Build Directory

The `build/` folder is where those native projects (e.g., VkFFT for GPU FFTs, glslang for shader compilation) are configured and compiled. Outputs here are then linked into Rust crates via `build.rs`, `cc`, `bindgen`, or consumed at runtime by feature layers.

This directory contains build system artifacts produced while integrating Candle with native GPU/CPU components. It exists alongside Cargo builds because some Candle features rely on C/C++ code and external toolchains (CMake/Ninja) to enable high‑performance tensor operations.

## Purpose in the Candle stack (Rust + tensors)

- Candle is primarily Rust, built with Cargo. For certain tensor ops (notably FFT on GPU via Vulkan), Candle integrates third‑party native projects that are better served by established C/C++ ecosystems.
- The `build/` folder is where those native projects are configured and compiled. Outputs here are then linked into Rust crates via `build.rs`, `cc`, `bindgen`, or consumed at runtime by feature layers.
- Key functionality this enables:
  - GPU FFTs using the VkFFT library (Vulkan backend). VkFFT is a high‑performance FFT implementation designed for GPUs; it compiles compute shaders at runtime.
  - Shader compilation support via the glslang toolchain, used to translate GLSL to SPIR‑V for Vulkan.
  - CPU remains fully supported by Candle’s pure‑Rust kernels; the native build path augments GPU acceleration, not basic CPU execution.

## What you’ll find here

- CMake cache and export files (e.g., `CMakeCache.txt`, `CMakeFiles/`, `glslang-main/*`).
- Ninja files if using Ninja generators (e.g., `build.ninja`).
- Exported CMake targets for glslang (see below), static libraries (`.a`) and tools (`glslang`, `spirv-remap`) laid out under the install prefix.
- `compile_commands.json` for C/C++ tooling and diagnostics.

These artifacts are generated; you normally do not edit them. They exist to make Rust<->C/C++ integration deterministic and reproducible during development.

## CPU vs GPU in Candle terms

- CPU: Candle’s tensor ops (e.g., matmul, convolutions, reductions) run in pure Rust by default; FFT may have a Rust/CPU path as well. No CMake involvement is required for the CPU‑only flow.
- GPU (Vulkan path): When enabling GPU FFT features, Candle uses VkFFT. VkFFT expects a GLSL→SPIR‑V compiler available at build/runtime; we vendor/build the Khronos glslang components to satisfy this. That’s why you see a `glslang-main` subtree here.
- GPU (CUDA path): Separately, CUDA features may use NVIDIA toolchains (e.g., cuFFT or custom kernels). Those are independent from glslang/Vulkan and may live in other build outputs. This folder can still host auxiliary CMake/Ninja outputs for such integrations.

## What “glslang-targets” means

Inside `glslang-main/CMakeFiles/Export/.../` you’ll find files like `glslang-targets.cmake` and `glslang-targets-debug.cmake`. These are CMake “exported targets” files: they describe how other CMake projects (or our build scripts) can import glslang libraries and tools.

Example from `glslang-targets-debug.cmake`:

```
IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libOSDependent.a"
```

- `glslang::OSDependent` is an imported target defined by the glslang build; `IMPORTED_LOCATION_DEBUG` points to the static library that should be linked when using Debug configuration.
- `${_IMPORT_PREFIX}` resolves to the install prefix for the exported package (typically `${CMAKE_INSTALL_PREFIX}` during packaging). The file also lists other targets such as:
  - `glslang::glslang`, `glslang::MachineIndependent`, `glslang::GenericCodeGen`, `glslang::SPIRV`, `glslang::HLSL`, `glslang::OGLCompiler`, `glslang::SPVRemapper`
  - Executables: `glslang::glslang-standalone` (tool), `glslang::spirv-remap`
- In Rust builds, we generally don’t consume these targets directly with CMake, but our native build steps rely on them to locate the correct libraries and tools so Rust crates can link/use Vulkan shader compilation via glslang.

## How this ties back to Candle crates

- Feature flags and layers related to GPU FFT (e.g., VkFFT/Vulkan) rely on the artifacts here to compile kernels and dispatch tensor FFTs on GPU.
- Crates that bridge to native code (e.g., GPU kernels or FFI layers) can use these outputs during their `build.rs` to find headers, libraries, and tooling.
- This keeps high‑performance paths (FFT, shader compilation) fast and portable while the high‑level tensor APIs remain idiomatic Rust.

## When you should care

- You’re enabling GPU FFT/Vulkan features and need glslang/VkFFT to be present and correctly built.
- You are debugging shader compilation, link errors to `libglslang.a`/`libSPIRV.a`, or missing tools like `glslang`.
- You are updating or pinning the third‑party stack under `third_party/VkFFT`.

## Notes

- These files are generated; regenerate by reconfiguring/rebuilding the native components (CMake/Ninja). Don’t hand‑edit exported target files.
- Keep host toolchains consistent (compiler, stdlib, libstdc++/libc++ versions) to avoid ABI/linking issues when Rust links against these `.a` files.
- If you only use CPU Candle, you can ignore this directory.