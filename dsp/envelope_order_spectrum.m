function P = envelope_order_spectrum(env, fs, fr, baseOrders, cfg)
L = cfg.phys.L;
e = env(:) - mean(env);
n = numel(e);
nfft = 2^nextpow2(n * cfg.phys.zeroPad);
E = abs(fft(e .* hann(n), nfft)) / n;
E = E(1:nfft/2);
f = (0:nfft/2-1).' * fs / nfft;

C = numel(baseOrders);
P = zeros(L, C, 'single');
uEdges = linspace(cfg.phys.umin, cfg.phys.umax, L + 1);
uCent  = (uEdges(1:end-1) + uEdges(2:end)) / 2;
for c = 1:C
    fEdges = uEdges * baseOrders(c) * fr;
    bin = discretize(f, fEdges);
    ok = ~isnan(bin);
    s = accumarray(bin(ok), E(ok), [L 1], @max, NaN);
    if any(isnan(s))
        s(isnan(s)) = interp1(f, E, uCent(isnan(s)) * baseOrders(c) * fr, 'linear', 0);
    end
    s = s / (median(s) + eps);
    P(:, c) = single(log1p(s));
end
end
