#![cfg(feature = "cuda")]

use candle_core::{Device, Result, Tensor};

fn cpu_dft_real(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut re = vec![0f32; n];
    let mut im = vec![0f32; n];
    let two_pi = std::f32::consts::TAU; // 2π
    for k in 0..n {
        let kf = k as f32;
        let mut rk = 0f32;
        let mut ik = 0f32;
        for (n_idx, &xn) in x.iter().enumerate() {
            let ang = two_pi * (kf * (n_idx as f32)) / (n as f32);
            rk += xn * ang.cos();
            ik -= xn * ang.sin(); // e^{-iθ}
        }
        re[k] = rk;
        im[k] = ik;
    }
    (re, im)
}

#[test]
fn gpu_dft_via_gemm_matches_cpu_small_n() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let n: usize = 16;

    // Random real signal on GPU
    let x = Tensor::randn(0f32, 1f32, (n,), &dev)?;

    // Build outer-product grid kn on GPU: shape (n,n)
    let k = Tensor::arange(0f32, n as f32, &dev)?.reshape((n, 1))?; // (n,1)
    let nn = Tensor::arange(0f32, n as f32, &dev)?.reshape((1, n))?; // (1,n)
    let kn = k.matmul(&nn)?; // (n,n)

    // omega = 2π/N * kn
    let factor = (std::f32::consts::TAU) / (n as f32);
    let factor_t = Tensor::full(factor, kn.shape(), &dev)?;
    let omega = kn.broadcast_mul(&factor_t)?;

    // Compute cos/sin on GPU
    let cos_w = omega.cos()?; // (n,n)
    let sin_w = omega.sin()?; // (n,n)

    // DFT via GEMM: re = cos_w @ x, im = -sin_w @ x
    let x_col = x.reshape((n, 1))?; // (n,1)
    let re = cos_w.matmul(&x_col)?.reshape((n,))?;
    let im = sin_w.neg()?.matmul(&x_col)?.reshape((n,))?;

    // Bring GPU results and CPU baseline for comparison
    let re_gpu = re.to_vec1::<f32>()?;
    let im_gpu = im.to_vec1::<f32>()?;

    let x_cpu = x.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let (re_cpu, im_cpu) = cpu_dft_real(&x_cpu);

    // Compare with a modest tolerance (O(N^2) dft vs f32 trig)
    let tol = 1e-2f32;
    for i in 0..n {
        assert!((re_gpu[i] - re_cpu[i]).abs() <= tol * (1.0 + re_cpu[i].abs()), "re[{i}] mismatch: gpu={} cpu={}", re_gpu[i], re_cpu[i]);
        assert!((im_gpu[i] - im_cpu[i]).abs() <= tol * (1.0 + im_cpu[i].abs()), "im[{i}] mismatch: gpu={} cpu={}", im_gpu[i], im_cpu[i]);
    }

    Ok(())
}
