clc
clear

version -blas
version -lapack

state_dim = 200;
num_steps = 10;

F = eye(state_dim) * 0.1;
Q = eye(state_dim);
H = ones(1, state_dim);
R = 0.1;
m0 = zeros(state_dim, 1);
p0 = eye(state_dim);

ys = randn(num_steps, 1);

mfs = zeros(num_steps, state_dim);
pfs = zeros(num_steps, state_dim, state_dim);

m = m0;
p = p0;

num_runs = 200;
burn_in = 20;
times = zeros(num_runs, 1);

for j = 1:num_runs
    tic
    for k = 1:num_steps
        m = F * m;
        p = F * p * F' + Q;

        S = H * p * H' + R;
        K = p * H' / S;
        m = m + K * (ys(k) - H * m);
        p = p - K * K' * S;

        mfs(k, :) = m;
        pfs(k, :, :) = p;
    end
    times(j) = toc;
end

times = times(burn_in:end);

fprintf("Mean %e, std %e, min %e, max %e \n", mean(times), std(times), min(times), max(times));
