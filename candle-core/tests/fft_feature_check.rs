//! Feature gate validation test to ensure FFT tests are run with proper feature flags enabled.
//! Provides helpful error messages and guidance when FFT feature is not enabled during testing.

//! Feature check test to ensure users know about the FFT feature requirement

#[test]
#[cfg(not(feature = "fft"))]
fn fft_feature_not_enabled() {
    panic!(
        "\n\n🚨 FFT Feature Required! 🚨\n\
        \n\
        The FFT tests require the 'fft' feature to be enabled.\n\
        \n\
        To run FFT tests, use:\n\
        ┌─────────────────────────────────────────────┐\n\
        │  cargo test --features fft                  │\n\
        │  cargo test --test fft_tests --features fft │\n\
        └─────────────────────────────────────────────┘\n\
        \n\
        This feature enables RustFFT integration for CPU-based FFT operations.\n\
        Without it, FFT operations will return a helpful error message.\n\
        \n"
    );
}

#[test]
#[cfg(feature = "fft")]
fn fft_feature_enabled_confirmation() {
    println!("✅ FFT feature is enabled - FFT tests can run!");
    // This test always passes when the feature is enabled
}
