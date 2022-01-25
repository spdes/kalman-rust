use ndarray::{concatenate, s, Array, Axis, Ix1};
use plotly::Scatter;

/// Equivalence to matplotlib.fill_between
/// 
/// Parameters
/// ----------
/// x : &Array<f64, Ix1>
///     The x coordinates.
/// y1 : &Array<f64, Ix1>
///     Upper y coordinates.
/// y2 : &Array<f64, Ix1>
///     Lower y coordinates.
/// 
/// Returns
/// -------
/// Box<Scatter<f64, f64>>
///     A Plotly Box object.
pub fn fill_between(
    x: &Array<f64, Ix1>,
    y1: &Array<f64, Ix1>,
    y2: &Array<f64, Ix1>,
) -> Box<Scatter<f64, f64>> {
    let stacked_xs = concatenate(Axis(0), &[x.clone().view(), x.slice(s![..;-1])]).unwrap();
    let stacked_ys = concatenate(Axis(0), &[y1.view(), y2.slice(s![..;-1])]).unwrap();
    Scatter::from_array(stacked_xs, stacked_ys)
}
