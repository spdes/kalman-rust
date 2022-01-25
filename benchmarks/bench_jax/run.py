import timeit
import numpy as np
import jax.numpy as jnp
import jax
from jax.config import config

config.update("jax_enable_x64", True)


def kf(F: jnp.ndarray, Q: jnp.ndarray,
       H: jnp.ndarray, R: float,
       ys: jnp.ndarray,
       m0: jnp.ndarray, p0: jnp.ndarray):
    def scan_body(carry, elem):
        m, p = carry
        y = elem

        m = F @ m
        p = F @ p @ F.T + Q

        S = H @ p @ H.T + R
        K = p @ H.T / S
        m = m + K * (y - H @ m)
        p = p - jnp.outer(K, K * S)
        return (m, p), (m, p)

    _, (mfs, pfs) = jax.lax.scan(scan_body, (m0, p0), ys)
    return mfs, pfs


def ks(mfs, Pfs, F, Q):
    def scan_body(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        mp = F @ mf
        Pp = F @ Pf @ F.T + Q

        c, low = jax.scipy.linalg.cho_factor(Pp)
        G = Pf @ jax.scipy.linalg.cho_solve((c, low), F).T
        ms = mf + G @ (ms - mp)
        Ps = Pf + G @ (Ps - Pp) @ G.T
        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = jax.lax.scan(scan_body, (mfs[-1], Pfs[-1]),
                                 (mfs[:-1], Pfs[:-1]), reverse=True)
    return mss, Pss


if __name__ == '__main__':
    num_dim = 200
    num_steps = 10

    F = jnp.eye(num_dim) * 0.1
    Q = jnp.eye(num_dim)
    H = jnp.ones((num_dim,))
    R = 0.1
    m0 = jnp.zeros((num_dim,))
    P0 = jnp.eye(num_dim)
    key = jax.random.PRNGKey(666)
    ys = jax.random.normal(shape=(num_steps,), key=key)


    @jax.jit
    def jitted_kf(data):
        return kf(F, Q, H, R, data, m0, P0)


    def test_kf():
        mfs, pfs = jitted_kf(ys)
        return np.asarray(mfs), np.asarray(pfs)


    pseudo_mfs = jnp.zeros((num_steps, num_dim))
    pseudo_pfs = jnp.tile(jnp.eye(num_dim), [num_steps, 1, 1])


    @jax.jit
    def jitted_ks(mfs, Pfs):
        return ks(mfs, Pfs, F, Q)

    def test_ks():
        mss, pss = jitted_ks(pseudo_mfs, pseudo_pfs)
        return np.asarray(mss), np.asarray(pss)


    # Trigger jit
    test_kf()
    test_ks()

    number_runs = 100
    repeat = 5

    print(jax.default_backend())

    ts = jnp.array(timeit.repeat("test_kf()", globals=globals(), number=number_runs, repeat=repeat)) / number_runs
    print(f"Mean {ts.mean()}, std {ts.std()}, min {ts.min()}, max {ts.max()}")

    ts = jnp.array(timeit.repeat("test_ks()", globals=globals(), number=number_runs, repeat=repeat)) / number_runs
    print(f"Mean {ts.mean()}, std {ts.std()}, min {ts.min()}, max {ts.max()}")
