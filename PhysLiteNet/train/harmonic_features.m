function F = harmonic_features(X, cfg)
[~, L, C, N] = size(X);
H = cfg.phys.nHarm;
e = linspace(cfg.phys.umin, cfg.phys.umax, L + 1);
u = (e(1:end-1) + e(2:end)) / 2;
F = zeros(N, C * H + 2, 'single');
for c = 1:C
    Xc = reshape(X(1, :, c, :), L, N);
    for h = 1:H
        m = abs(u - h) <= 0.03 * h + 0.02;
        F(:, (c - 1) * H + h) = max(Xc(m, :), [], 1).';
    end
end
iO = find(cfg.phys.baseOrders == "BPFO"); iI = find(cfg.phys.baseOrders == "BPFI");
sO = sum(F(:, (iO - 1) * H + (1:H)), 2); sI = sum(F(:, (iI - 1) * H + (1:H)), 2);
F(:, C * H + 1) = sO ./ (sI + eps);
F(:, C * H + 2) = max(F(:, (iO - 1) * H + 1), F(:, (iI - 1) * H + 1));
end
