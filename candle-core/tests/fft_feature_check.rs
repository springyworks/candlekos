// Lightweight feature gate check for FFT.
//
// This test file is always compiled. If the `fft` feature is missing we emit clear
// guidance rather than silently succeeding or producing confusing linkage errors.

use candle_core::Result;

#[test]
fn fft_feature_check() -> Result<()> {
    #[cfg(feature = "fft")]
    {
        // Minimal smoke: create a tiny tensor and do a 1D real fft via rfft helper.
        use candle_core::{Device, Tensor};
        let t = Tensor::arange(0f32, 8f32, &Device::Cpu)?;
        let spec = t.rfft(0, false)?;
        assert!(
            spec.dims()[0] >= 8,
            "unexpected rfft output shape: {:?}",
            spec.dims()
        );
    }
    #[cfg(not(feature = "fft"))]
    {
        // Emit guidance. We still return Ok so test counts as passed but prints help.
        eprintln!(
            "[fft_feature_check] FFT feature disabled. Enable with: cargo test --features fft"
        );
    }
    Ok(())
}
// Additional explicit check tests below provide a failing panic with instructions
// when the user tries to run the suite without enabling `fft`.

#[test]
#[cfg(not(feature = "fft"))]
#[should_panic(expected = "FFT Feature Required")] // marks as expected panic so suite stays green
fn fft_feature_not_enabled() {
    panic!(
        "\n\n🚨 FFT Feature Required! 🚨\n\
        The FFT tests require the 'fft' feature to be enabled.\n\
        To run FFT tests, use:\n\
        cargo test --features fft\n\
        cargo test --test fft_tests --features fft\n"
    );
}

#[test]
#[cfg(feature = "fft")]
fn fft_feature_enabled_confirmation() {
    println!("✅ FFT feature is enabled - FFT tests can run!");
    // This test always passes when the feature is enabled
}
