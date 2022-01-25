import numpy as np
import timeit


def kf(F: np.ndarray, Q: np.ndarray,
       H: np.ndarray, R: float,
       ys: np.ndarray,
       m0: np.ndarray, p0: np.ndarray):
    dim_x = m0.size
    num_y = ys.size

    mfs = np.zeros(shape=(num_y, dim_x))
    pfs = np.zeros(shape=(num_y, dim_x, dim_x))

    m = m0
    p = p0

    # Filtering pass
    for k in range(num_y):
        # Pred
        m = F @ m
        p = F @ p @ F.T + Q

        # Update
        S = H @ p @ H.T + R
        K = p @ H.T / S
        m = m + K * (ys[k] - H @ m)
        p = p - np.outer(K, K) * S

        # Save
        mfs[k] = m
        pfs[k] = p
    return mfs, pfs


if __name__ == '__main__':
    
    num_dim = 200
    num_steps = 10
    
    np.random.seed(666)

    F = np.eye(num_dim) * 0.1
    Q = np.eye(num_dim)
    H = np.ones((num_dim,))
    R = 0.1
    m0 = np.zeros((num_dim,))
    P0 = np.eye(num_dim)
    ys = np.random.randn(num_steps)

    number_runs = 100
    repeat = 5
    
    np.show_config()
    
    ts = np.array(timeit.repeat("kf(F, Q, H, R, ys, m0, P0)", globals=globals(), number=number_runs, repeat=repeat)) / number_runs
    print(f"Mean {ts.mean()}, std {ts.std()}, min {ts.min()}, max {ts.max()}")
