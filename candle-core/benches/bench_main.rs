mod benchmarks;
use criterion::{criterion_main, criterion_group, Criterion};

fn all_benches(c: &mut Criterion) {
    benchmarks::affine::benches(c);
    benchmarks::copy::benches(c);
    benchmarks::conv_transpose2d::benches(c);
    benchmarks::matmul::benches(c);
    benchmarks::qmatmul::benches(c);
    benchmarks::random::benches(c);
    benchmarks::reduce::benches(c);
    benchmarks::unary::benches(c);
    benchmarks::where_cond::benches(c);
    benchmarks::fft::benches(c);
    #[cfg(feature = "fft")]
    { benchmarks::fft_large::benches(c); }
    #[cfg(feature = "fft")]
    { benchmarks::fft_ratio::benches(c); }
}

criterion_group!(core_benches, all_benches);
criterion_main!(core_benches);
