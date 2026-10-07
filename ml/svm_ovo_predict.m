function S = svm_ovo_predict(M, X)
X = (double(X) - M.mu) ./ M.sd;
K = numel(M.classes);
votes = zeros(size(X, 1), K);
marg  = zeros(size(X, 1), K);
for q = 1:size(M.pairs, 1)
    B = M.bin{q};
    d = sum(A_rbf(X, B.SV, M.gamma) .* B.coef.', 2) + B.b;
    a = M.pairs(q, 1); b = M.pairs(q, 2);
    votes(:, a) = votes(:, a) + (d > 0);
    votes(:, b) = votes(:, b) + (d <= 0);
    marg(:, a) = marg(:, a) + d;
    marg(:, b) = marg(:, b) - d;
end
S = votes + 1e-3 * tanh(marg);
S = S - min(S, [], 2) + eps;
S = S ./ sum(S, 2);
end

function Kmat = A_rbf(A, B, gamma)
d = sum(A.^2, 2) + sum(B.^2, 2).' - 2 * (A * B.');
Kmat = exp(-gamma * max(d, 0));
end
