#![cfg(feature = "fft")]
use criterion::{black_box, Criterion};
use candle_core::{Device, Tensor, Result};
use super::{BenchDevice, BenchDeviceHandler};

fn bench_rfft(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for dev in handler.devices.into_iter() {
        let name = dev.bench_name("rfft_1d_4096");
        c.bench_function(&name, |b| {
            // 1D real input length 4096
            let t = Tensor::randn(0f32, 1.0, (4096,), &dev).unwrap();
            b.iter(|| {
                let f = t.rfft(0, false).unwrap();
                black_box(&f);
            });
        });
    }
}

fn bench_fft_complex(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for dev in handler.devices.into_iter() {
        let name = dev.bench_name("cfft_1d_2048");
        c.bench_function(&name, |b| {
            // Interleaved complex length 2048 -> shape (4096,)
            let t = Tensor::randn(0f32, 1.0, (2048*2,), &dev).unwrap();
            b.iter(|| {
                let f = t.fft(0, false, false).unwrap();
                black_box(&f);
            });
        });
    }
}

pub fn benches(c: &mut Criterion) {
    bench_rfft(c);
    bench_fft_complex(c);
}
