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

    build.compile("candle_vkfft_wrapper");

    // Link with CUDA runtime and driver libraries used by VkFFT CUDA backend.
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
