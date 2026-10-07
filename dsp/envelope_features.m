function [f, names] = envelope_features(P, xb, env, cfg)
L = cfg.phys.L; H = cfg.phys.nHarm;
u = linspace(cfg.phys.umin, cfg.phys.umax, L + 1);
u = (u(1:end-1) + u(2:end)) / 2;
chName = cfg.phys.baseOrders;

f = []; names = strings(0);
for c = 1:size(P, 2)
    for h = 1:H
        tol = 0.03 * h + 0.02;
        m = abs(u - h) <= tol;
        f(end+1) = max(P(m, c));
        names(end+1) = sprintf('%s_H%d', chName(c), h);
    end
end
iO = find(chName == "BPFO"); iI = find(chName == "BPFI");
if ~isempty(iO) && ~isempty(iI)
    sO = sum(f((iO-1)*H + (1:H))); sI = sum(f((iI-1)*H + (1:H)));
    f(end+1) = sO / (sI + eps);            names(end+1) = "ratio_O_I";
    f(end+1) = max(f((iO-1)*H+1), f((iI-1)*H+1)); names(end+1) = "max_H1";
end
xb = xb(:); a = abs(xb); r = rms(xb);
f(end+1) = kurt(xb);                     names(end+1) = "kurtosis";
f(end+1) = skew(xb);                     names(end+1) = "skewness";
f(end+1) = max(a) / r;                   names(end+1) = "crest";
f(end+1) = max(a) / mean(a);             names(end+1) = "impulse";
f(end+1) = r / mean(a);                  names(end+1) = "shape";
f(end+1) = max(a) / mean(sqrt(a))^2;     names(end+1) = "clearance";
f(end+1) = kurt(env(:));                 names(end+1) = "env_kurtosis";
f = single(f);
end

function k = kurt(x)
x = x - mean(x); k = mean(x.^4) / (mean(x.^2)^2 + eps);
end

function s = skew(x)
x = x - mean(x); s = mean(x.^3) / (mean(x.^2)^1.5 + eps);
end
