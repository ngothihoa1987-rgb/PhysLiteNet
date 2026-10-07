function p = stat_signrank(x, y)
d = x(:) - y(:);
d = d(d ~= 0);
n = numel(d);
if n == 0, p = 1; return; end
r = rank_abs(abs(d));
W = sum(r(d > 0));
mu = n * (n + 1) / 4;
if n <= 20
    signs = dec2bin(0:2^n - 1) == '1';
    Wall = signs * r(:);
    p = min(1, 2 * min(mean(Wall <= W + 1e-9), mean(Wall >= W - 1e-9)));
else
    [~, ~, g] = unique(abs(d));
    t = accumarray(g, 1);
    s = sqrt(n * (n + 1) * (2 * n + 1) / 24 - sum(t.^3 - t) / 48);
    z = (abs(W - mu) - 0.5) / s;
    p = erfc(z / sqrt(2));
end
end

function r = rank_abs(x)
[xs, o] = sort(x);
r = zeros(size(x));
i = 1; n = numel(x);
while i <= n
    j = i;
    while j < n && xs(j + 1) == xs(i), j = j + 1; end
    r(o(i:j)) = (i + j) / 2;
    i = j + 1;
end
end
