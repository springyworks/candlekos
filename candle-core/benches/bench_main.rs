mod benchmarks;
use criterion::{Criterion, criterion_group, criterion_main};

fn all_benches(_c: &mut Criterion) {
    benchmarks::affine::benches();
    benchmarks::copy::benches();
    benchmarks::conv_transpose2d::benches();
    benchmarks::matmul::benches();
    benchmarks::qmatmul::benches();
    benchmarks::random::benches();
    benchmarks::reduce::benches();
    benchmarks::unary::benches();
    benchmarks::where_cond::benches();
    #[cfg(feature = "fft")]
    {
        benchmarks::fft::benches();
        benchmarks::fft_large::benches();
        benchmarks::fft_ratio::benches();
    }
}

criterion_group!(core_benches, all_benches);
criterion_main!(core_benches);
