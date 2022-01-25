using BenchmarkTools
using LinearAlgebra

function kf(F, Q, H, R, ys, m0, p0)

    dim_x = length(m0);
    num_y = length(ys);

    mfs = zeros(num_y, dim_x);
    pfs = zeros(num_y, dim_x, dim_x);

    m = m0;
    p = p0;

    for k in length(ys)
        m = F * m;
        p = F * p * F' + Q;

        S = H * p * H' .+ R;
        K = p * H' ./ S;
        m = m + K * (ys[k] .- H * m);
        p = p - (K * K') .* S;

        mfs[k, :] = m;
        pfs[k, :, :] = p;
    end

    return mfs, pfs
end

state_dim = 200;
num_steps = 10;

F = Matrix(I, state_dim, state_dim) * 0.1;
Q = Matrix(I, state_dim, state_dim);
H = ones(1, state_dim);
R = 0.1;
m0 = zeros(state_dim, 1);
p0 = Matrix(I, state_dim, state_dim);
ys = randn(num_steps, 1);

m = m0;
p = p0;

# The results seems only print in interactive terminal. don't know how to fix this
@benchmark kf(F, Q, H, R, ys, m0, p0) setup=(F, Q, H, R, ys, m0, p0);
