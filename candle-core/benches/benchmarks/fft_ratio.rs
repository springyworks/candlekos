#![cfg(feature = "fft")]
// Bench that (if both CPU + CUDA available) measures simple 1D RFFT throughput ratio and prints it.
// Enabled via CANDLE_FFT_RATIO=1 to avoid default noise.
use criterion::{Criterion, criterion_group};
use candle_core::{Device, Tensor};

fn ratio_bench(c: &mut Criterion) {
    if std::env::var("CANDLE_FFT_RATIO").ok().map(|v| v=="1" || v.eq_ignore_ascii_case("true")).unwrap_or(false) == false { return; }
    let cpu = Device::Cpu;
    let cuda = match Device::new_cuda(0) { Ok(d) => d, Err(_) => { return; } };
    let n = 8192usize;
    let input_cpu = Tensor::randn(0f32, 1.0, (n,), &cpu).unwrap();
    let input_gpu = input_cpu.to_device(&cuda).unwrap();
    let mut group = c.benchmark_group("fft_ratio_rfft1d_8192");
    group.bench_function("cpu", |b| {
        b.iter(|| { let f = input_cpu.rfft(0, false).unwrap(); criterion::black_box(f); });
    });
    group.bench_function("cuda", |b| {
        b.iter(|| { let f = input_gpu.rfft(0, false).unwrap(); criterion::black_box(f); });
    });
    group.finish();
}

fn fft_ratio_benches(c: &mut Criterion) { ratio_bench(c); }

criterion_group!(benches, fft_ratio_benches);
