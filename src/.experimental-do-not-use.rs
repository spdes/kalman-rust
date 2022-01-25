// EXPERIMENTAL DO NOT USE
// pub fn kf_scan(
//     m0: &Array<f64, Ix1>,
//     p0: &Array<f64, Ix2>,
//     F: &Array<f64, Ix2>,
//     Sigma: &Array<f64, Ix2>,
//     H: &Array<f64, Ix1>,
//     Xi: f64,
//     ys: &Array<f64, Ix1>,
// ) -> () {
//     let iter = ys.iter();

//     let init = vec![m0.clone(), p0.clone().into_shape((10000, )).unwrap()];

//     let results = iter.scan(init, |carry, elem| {

//         let mp = F.dot(&carry[0]);
//         let _temp = &carry[1].clone().into_shape((100, 100)).unwrap();
//         let pp = F.dot(_temp).dot(&(F.t())) + Sigma;

//         // Update
//         let S = H.dot(&pp).dot(H) + Xi;
//         let K = H.dot(&pp) / S;

//         let mf = &K * (elem - H.dot(&mp));
//         let pf = outer(&K, &(&K * S));

//         *carry = vec![mf.clone(), pf.clone().into_shape((10000, )).unwrap()];

//         Some(vec![mf, pf.into_shape((10000, )).unwrap().clone()])
//     });

//     let zz = Array::from_iter(results);
// }
