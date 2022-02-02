#![allow(non_snake_case)]

use ndarray as nd;
use ndarray::{s, Array, Ix1, Ix2, Ix3};
use ndarray_linalg::cholesky::UPLO;
use ndarray_linalg::{Cholesky, InverseC};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Distribution, Normal};
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha20Rng;

/// Outer product of two vectors
/// 
/// Parameters
/// ----------
/// x : &Array<f64, Ix1>
///     A vector.
/// y : &Array<f64, Ix1>
///     A vector.
/// 
/// Returns
/// -------
/// Array<f64, Ix2>
///     The outer product of x and y.
pub fn outer(x: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> Array<f64, Ix2> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_reshaped = x.view().into_shape((size_x, 1)).unwrap();
    let y_reshaped = y.view().into_shape((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

/// Make Matern 3/2 state-space model
/// 
/// Parameters
/// ----------
/// ell : f64
///     The length scale parameter of the Matern process.
/// sigma : f64
///     The magnitude scale parameter of the Matern process.
/// dt : f64
///     Time step (uniform) of discretisation.
/// 
/// Returns
/// -------
/// Three Array<f64, Ix2>
///     A tuple consisting of the state transition matrix, covariance matrix and its Cholesky decomposition.
pub fn make_matern32_ssm(
    ell: f64,
    sigma: f64,
    dt: f64,
) -> (Array<f64, Ix2>, Array<f64, Ix2>, Array<f64, Ix2>) {
    let lam = f64::powf(3.0, 1.0 / 2.0) / ell;
    let dt_lam = dt * lam;
    let beta = sigma.powi(2) * f64::exp(-2. * dt_lam);

    let F = nd::array![[1. + dt_lam, dt], [-dt * lam.powi(2), 1. - dt_lam]]
        .map(|x| x * (-dt_lam).exp());
    let Sigma = nd::array![
        [
            sigma.powi(2) - beta * (2. * dt_lam + 2. * dt_lam.powi(2) + 1.),
            2. * dt.powi(2) * lam.powi(3) * beta
        ],
        [
            2. * dt.powi(2) * lam.powi(3) * beta,
            lam.powi(2) * (sigma.powi(2) + beta * (2. * dt_lam - 2. * dt_lam.powi(2) - 1.))
        ]
    ];
    let chol = Sigma.cholesky(UPLO::Lower).unwrap();
    (F, chol, Sigma)
}

/// Simulate a trajectory and its measurements from an LTI state-space model
/// 
/// Parameters
/// ----------
/// m0 : &Array<f64, Ix1>
///     Initial mean.
/// p0 : &Array<f64, Ix2>
///     Initial covariance.
/// F : &Array<f64, Ix2>
///     State transition matrix.
/// chol_Sigma : &Array<f64, Ix2>
///     Cholesky (lower) decomposition of the state covariance.
/// H : &Array<f64, Ix1>
///     Measurement transformation vector.
/// Xi : f64
///     Measurement noise covariance.
/// num_steps : usize
///     Number of times.
/// 
/// Returns
/// -------
/// (Array<f64, Ix2>, Array<f64, Ix1>)
///     A trajectory of the state process and its measurements.
pub fn simulate_lti_ssm(
    m0: &Array<f64, Ix1>,
    p0: &Array<f64, Ix2>,
    F: &Array<f64, Ix2>,
    chol_Sigma: &Array<f64, Ix2>,
    H: &Array<f64, Ix1>,
    Xi: f64,
    num_steps: usize,
) -> (Array<f64, Ix2>, Array<f64, Ix1>) {
    let state_dim = m0.shape()[0];
    let mut xs = Array::<f64, Ix2>::zeros((num_steps, state_dim));
    let mut ys = Array::<f64, Ix1>::zeros((num_steps, ));

    let mut rng_key = ChaCha20Rng::seed_from_u64(666);
    let normal_distribution = Normal::new(0., 1.).unwrap();

    // Generate x0
    let chol_p0 = p0.cholesky(UPLO::Lower).unwrap();
    let mut x = m0
        + chol_p0.dot(
        &(Array::<f64, Ix1>::random_using((state_dim, ), normal_distribution, &mut rng_key)),
    );

    // Loop
    for k in 0..num_steps {
        let rand_x =
            Array::<f64, Ix1>::random_using((state_dim, ), normal_distribution, &mut rng_key);
        x = F.dot(&x) + chol_Sigma.dot(&rand_x);

        let rand_y = normal_distribution.sample(&mut rng_key);
        let y = H.dot(&x) + rand_y * Xi.sqrt();

        ys[k] = y;
        xs.slice_mut(s![k, ..]).assign(&x);
        // xs.index_axis_mut(Axis(0), k).assign(&x);
    }
    (xs, ys)
}

/// Kalman filter
/// 
/// Parameters
/// ----------
/// m0 : &Array<f64, Ix1>
///     Initial mean.
/// p0 : &Array<f64, Ix2>
///     Initial covariance.
/// F : &Array<f64, Ix2>
///     State transition matrix.
/// Sigma : &Array<f64, Ix2>
///     State covariance.
/// H : &Array<f64, Ix1>
///     Measurement transformation vector.
/// Xi : f64
///     Measurement noise covariance.
/// ys : &Array<f64, Ix1>
///     Measurements.
/// 
/// Returns
/// -------
/// (Array<f64, Ix2>, Array<f64, Ix3>)
///     Filtering means and covariances.
pub fn kf(
    m0: &Array<f64, Ix1>,
    p0: &Array<f64, Ix2>,
    F: &Array<f64, Ix2>,
    Sigma: &Array<f64, Ix2>,
    H: &Array<f64, Ix1>,
    Xi: f64,
    ys: &Array<f64, Ix1>,
) -> (Array<f64, Ix2>, Array<f64, Ix3>) {
    let (n_measurements, state_dim) = (ys.shape()[0], m0.shape()[0]);

    let mut mfs = Array::<f64, Ix2>::zeros((n_measurements, state_dim));
    let mut pfs = Array::<f64, Ix3>::zeros((n_measurements, state_dim, state_dim));

    let mut m = m0.clone();
    let mut p = p0.clone();

    // Loop
    for k in 0..n_measurements {
        // Prediction
        m = F.dot(&m);
        p = F.dot(&p).dot(&F.t()) + Sigma;

        // Update
        let S = H.dot(&p).dot(H) + Xi;
        let K = H.dot(&p) / S;

        m += &(&K * (ys[k] - H.dot(&m)));
        p -= &(outer(&K, &(&K * S)));

        // Save
        mfs.slice_mut(s![k, ..]).assign(&m);
        pfs.slice_mut(s![k, .., ..]).assign(&p);

        // In principle you could also do the assignment in parallel as in the below.
        // But it becomes even slower, I don't know why. 
        // Also recall to enable "rayon" feature of you want to try this.
        // Zip::from(&mut mfs.slice_mut(s![k, ..])).and(&m).par_for_each(|a, &b| *a = b);
        // Zip::from(&mut pfs.slice_mut(s![k, .., ..])).and(&p).par_for_each(|a, &b| *a = b);
    }
    (mfs, pfs)
}

/// RTS smoother
/// 
/// Parameters
/// ----------
/// mfs : &Array<f64, Ix2>
///     Filtering means.
/// pfs : &Array<f64, Ix3>
///     Filtering covariances
/// F : &Array<f64, Ix2>
///     State transition matrix.
/// Sigma : &Array<f64, Ix2>
///     State covariance.
/// 
/// Returns
/// -------
/// (Array<f64, Ix2>, Array<f64, Ix3>)
///     Smoothing means and covariances.
pub fn ks(
    mfs: &Array<f64, Ix2>,
    pfs: &Array<f64, Ix3>,
    F: &Array<f64, Ix2>,
    Sigma: &Array<f64, Ix2>,
) -> (Array<f64, Ix2>, Array<f64, Ix3>) {
    let mut mss = mfs.clone();
    let mut pss = pfs.clone();

    let num_measurements = mss.shape()[0];

    for k in (0..num_measurements - 1).rev() {
        let mf = mfs.slice(s![k, ..]);
        let pf = pfs.slice(s![k, .., ..]);

        let mp = F.dot(&mf);
        let pp = F.dot(&pf).dot(&F.t()) + Sigma;

        // ndarray-linalg does not support .solve() for matrix input yet, so 
        // I have to use Cholesky inverse .invc() here.
        let gain = pf.dot(&F.t()).dot(&pp.invc().unwrap());

        let ms = &mf + gain.dot(&(&mss.slice(s![k + 1, ..]) - mp));
        let ps = &pf
            + gain
            .dot(&(&pss.slice(s![k + 1, .., ..]) - pp))
            .dot(&gain.t());

        mss.slice_mut(s![k, ..]).assign(&ms);
        pss.slice_mut(s![k, .., ..]).assign(&ps);
    }
    (mss, pss)
}

