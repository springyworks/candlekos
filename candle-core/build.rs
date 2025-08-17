fn main() {
    // Only run when VkFFT FFI and CUDA features are enabled for candle-core.
    let use_vkfft = std::env::var_os("CARGO_FEATURE_GPU_FFT_VKFFT_FFI").is_some();
    let use_cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    if !(use_vkfft && use_cuda) {
        return;
    }

    println!("cargo:rerun-if-changed=src/cuda_backend/cuda_fft/vkfft_wrapper.c");
    println!("cargo:rerun-if-changed=../third_party/VkFFT/vkFFT/vkFFT.h");

    let mut build = cc::Build::new();
    build.file("src/cuda_backend/cuda_fft/vkfft_wrapper.c");
    // Include the vendored VkFFT directory roots so nested headers resolve.
    build.include("../third_party/VkFFT");
    build.include("../third_party/VkFFT/vkFFT");
    // Select CUDA backend in VkFFT headers.
    build.define("VKFFT_BACKEND", Some("1")); // 1 = CUDA
    // Be permissive about C standard to accommodate varied toolchains.
    build.flag_if_supported("-std=c11");

    // Ultra-quiet option: when CANDLE_QUIET=1, suppress warnings from third-party C code.
    if std::env::var_os("CANDLE_QUIET").as_deref() == Some(std::ffi::OsStr::new("1")) {
        // Disable warnings in cc and pass common suppression flags when supported.
        build.warnings(false);
        if !cfg!(target_env = "msvc") {
            // GCC/Clang style flags
            build.flag_if_supported("-w");
            build.flag_if_supported("-Wno-unused-parameter");
            build.flag_if_supported("-Wno-unused-variable");
            build.flag_if_supported("-Wno-sign-compare");
            build.flag_if_supported("-Wno-implicit-fallthrough");
        }
    }

    build.compile("candle_vkfft_wrapper");

    // Link with CUDA runtime and driver libraries used by VkFFT CUDA backend.
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
