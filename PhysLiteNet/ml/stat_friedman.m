function [p, chi2, meanRank] = stat_friedman(M)
[b, k] = size(M);
R = zeros(b, k);
for i = 1:b, R(i, :) = rank_ties(-M(i, :)); end
meanRank = mean(R, 1);
chi2 = 12 * b / (k * (k + 1)) * sum((meanRank - (k + 1) / 2).^2);
tieSum = 0;
for i = 1:b
    [~, ~, g] = unique(M(i, :));
    t = accumarray(g(:), 1);
    tieSum = tieSum + sum(t.^3 - t);
end
chi2 = chi2 / (1 - tieSum / (b * k * (k^2 - 1)));
p = gammainc(chi2 / 2, (k - 1) / 2, 'upper');
end

function r = rank_ties(x)
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
