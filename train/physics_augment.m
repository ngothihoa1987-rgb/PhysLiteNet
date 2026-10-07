function X = physics_augment(X, cfg, baseOrders)
A = cfg.aug;
[~, L, C, B] = size(X);
e = linspace(cfg.phys.umin, cfg.phys.umax, L + 1);
u = ((e(1:end-1) + e(2:end)) / 2).';
du = u(2) - u(1);
sig = A.peakWidthBins * du;
X = double(X);
for b = 1:B
    S = expm1(reshape(X(1, :, :, b), L, C));
    if rand < A.pJitter
        a = 1 + A.jitter * (2 * rand - 1);
        for c = 1:C
            S(:, c) = interp1(u, S(:, c), u / a, 'linear', median(S(:, c)));
        end
    end
    orders = []; amps = [];
    if rand < A.pShaftHarm
        k = find(rand(1, A.maxShaftHarm) < 0.5);
        if isempty(k), k = 1; end
        orders = [orders, k];
        amps = [amps, A.ampShaft(1) + diff(A.ampShaft) * rand(1, numel(k)) ./ sqrt(k)];
    end
    if rand < A.pSpur
        n = randi(A.nSpur);
        maxOrder = cfg.phys.umax * max(baseOrders);
        orders = [orders, 0.3 + (maxOrder - 0.3) * rand(1, n)];
        amps = [amps, A.ampSpur(1) + diff(A.ampSpur) * rand(1, n)];
    end
    if ~isempty(orders)
        for c = 1:C
            pos = orders / baseOrders(c);
            S(:, c) = S(:, c) + sum(amps .* exp(-(u - pos).^2 / (2 * sig^2)), 2);
        end
    end
    if A.noise > 0
        S = S .* max(0, 1 + A.noise * randn(L, C));
    end
    S = S ./ (median(S, 1) + eps);
    X(1, :, :, b) = reshape(log1p(S), 1, L, C);
end
X = single(X);
end
