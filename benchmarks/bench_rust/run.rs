#![allow(non_snake_case)]

use kalman_rust::kfs::{kf, ks};
use ndarray::{Array, Ix1, Ix2, Ix3};

use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {

    // Model parameters and construction
    // Change the state dimension to see speed differences
    
    let state_dim: usize = 200;
    let num_steps: usize = 10;

    let F = Array::<f64, Ix2>::eye(state_dim) * 0.1;
    let Sigma = Array::<f64, Ix2>::eye(state_dim);
    let H = Array::<f64, Ix1>::ones((state_dim,));
    let Xi: f64 = 0.1;

    let m0 = Array::<f64, Ix1>::ones((state_dim,));
    let p0 = Array::<f64, Ix2>::eye(state_dim);
    let ys = Array::<f64, Ix1>::ones((num_steps,));

    c.bench_function("Benchmarking Kalman filtering", |b| {
        b.iter(|| kf(&m0, &p0, &F, &Sigma, &H, Xi, &ys))
    });

    let mfs = Array::<f64, Ix2>::zeros((num_steps, state_dim));
    let pfs = Array::<f64, Ix3>::zeros((num_steps, state_dim, state_dim)) + Array::<f64, Ix2>::eye(state_dim);

    c.bench_function("Benchmarking RTS smoothing", |b| {
        b.iter(|| ks(&mfs, &pfs, &F, &Sigma))
    });
}

criterion_group!(bench_rust, criterion_benchmark);
criterion_main!(bench_rust);
