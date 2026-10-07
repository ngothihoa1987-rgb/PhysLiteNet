function [T, W] = harmonic_template(Ls, cfg)
e = linspace(cfg.phys.umin, cfg.phys.umax, Ls + 1);
u = (e(1:end-1) + e(2:end)).' / 2;
H = cfg.phys.nHarm;
W = zeros(Ls, H);
for h = 1:H
    sig = 0.03 * h + 0.03;
    W(:, h) = exp(-(u - h).^2 / (2 * sig^2));
end
T = sum(W, 2).';
W = W ./ sum(W, 1);
end
