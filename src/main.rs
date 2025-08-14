fn main() {
    println!("candle workspace entry point\n");
    println!("Common actions:\n  cargo test            -> run default feature tests (lean)\n  cargo test --features fft         -> include FFT tests\n  cargo bench --features fft        -> 1D FFT benches\n  CANDLE_FFT_LARGE=1 cargo bench --features fft  -> add 2D benches\n  CANDLE_FFT_RATIO=1 cargo bench --features fft,cuda,gpu-fft -> CPU vs GPU ratio\n  CANDLE_FFT_DEBUG=1 cargo test --features fft-debug,fft -> enable debug macro output\n");
    println!("Use xtask for matrix/powerset: cargo run -p xtask -- help");
}
