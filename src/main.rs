#![allow(non_snake_case)]

use ndarray as nd;
use ndarray::{s, Array, Ix1};

use plotly::common::color::NamedColor::{Black, Transparent};
use plotly::common::{Fill, Line, Mode, Title};
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};

use kalman_rust::kfs::{kf, ks, make_matern32_ssm, simulate_lti_ssm};
use kalman_rust::utils::fill_between;

fn main() {
    // Time step and instances
    let dt: f64 = 0.01;
    let num_steps: usize = 1000;
    let ts = Array::<f64, Ix1>::linspace(dt, dt * (num_steps as f64), num_steps);

    // State-space Mmdel parameters and construction
    let ell: f64 = 0.5;
    let sigma: f64 = 1.;
    let (F, chol_Sigma, Sigma) = make_matern32_ssm(ell, sigma, dt);
    let H = nd::array![1., 0.];
    let Xi: f64 = 0.1;

    let m0 = Array::<f64, Ix1>::zeros((2,));
    let p0 = nd::array![
        [sigma.powi(2), 0.],
        [0., (f64::sqrt(3.) / ell).powi(2) * sigma.powi(2)]
    ];

    // Simulate a trajectory from the SSM and its measurements
    let (xs, ys) = simulate_lti_ssm(&m0, &p0, &F, &chol_Sigma, &H, Xi, num_steps);

    // Run Kalman filtering
    let (mfs, pfs) = kf(&m0, &p0, &F, &Sigma, &H, Xi, &ys);

    // Run RTS smoothing
    let (mss, pss) = ks(&mfs, &pfs, &F, &Sigma);

    // Plot
    let trace_x = Scatter::from_array(ts.clone(), xs.slice(s![.., 0]).to_owned())
        .mode(Mode::Lines)
        .name("True trajectory");
    let trace_y = Scatter::from_array(ts.clone(), ys)
        .mode(Mode::Markers)
        .name("Measurements");
    let trace_mf = Scatter::from_array(ts.clone(), mfs.slice(s![.., 0]).to_owned())
        .mode(Mode::Lines)
        .name("Filtering mean");
    let trace_ms = Scatter::from_array(ts.clone(), mss.slice(s![.., 0]).to_owned())
        .mode(Mode::Lines)
        .name("Smoothing mean");

    let conf_upper = &mss.slice(s![.., 0]) + pss.slice(s![.., 0, 0]).map(|x| (*x).sqrt() * 1.96);
    let conf_lower = &mss.slice(s![.., 0]) - pss.slice(s![.., 0, 0]).map(|x| (*x).sqrt() * 1.96);
    let trace_area = fill_between(&ts, &conf_upper, &conf_lower)
        .fill(Fill::ToSelf)
        .fill_color(Black)
        .opacity(0.15)
        .line(Line::new().color(Transparent))
        .name("Smoothing .95 Confidence");

    let mut plt = Plot::new();

    let layout = Layout::new()
        .title(Title::new("Kalman filtering and RTS smoothing"))
        .x_axis(Axis::new().title(Title::new("time (t)")));

    plt.set_layout(layout);
    plt.add_trace(trace_x);
    plt.add_trace(trace_y);
    plt.add_trace(trace_mf);
    plt.add_trace(trace_ms);
    plt.add_trace(trace_area);
    plt.show();
}
