#![cfg(feature = "fft")]
// Optional larger multi-dimensional FFT benchmarks gated by env var to avoid default CI cost.
// Enable via: CANDLE_FFT_LARGE=1 cargo bench --features fft[,cuda,gpu-fft]
use super::{BenchDevice, BenchDeviceHandler};
use candle_core::{Device, Tensor};
use criterion::{Criterion, black_box, criterion_group};

fn bench_2d(c: &mut Criterion) {
    if std::env::var("CANDLE_FFT_LARGE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        == false
    {
        return;
    }
    let handler = BenchDeviceHandler::new().unwrap();
    for dev in handler.devices.into_iter() {
        let name = dev.bench_name("rfft2_256x256");
        c.bench_function(&name, |b| {
            let t = Tensor::randn(0f32, 1.0, (256, 256), &dev).unwrap();
            b.iter(|| {
                let f = t.rfft2(false).unwrap();
                black_box(&f);
            });
        });
    }
}

fn fft_large_benches(c: &mut Criterion) {
    bench_2d(c);
}

criterion_group!(benches, fft_large_benches);
