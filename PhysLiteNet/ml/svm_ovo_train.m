function M = svm_ovo_train(X, Y, C, gamma)
if nargin < 3 || isempty(C), C = 1; end
X = double(X);
M.mu = mean(X, 1); M.sd = std(X, 0, 1) + eps;
X = (X - M.mu) ./ M.sd;
if nargin < 4 || isempty(gamma), gamma = 1 / size(X, 2); end
M.gamma = gamma; M.C = C;
M.classes = categories(Y);
K = numel(M.classes);
M.pairs = nchoosek(1:K, 2);
M.bin = cell(size(M.pairs, 1), 1);
for q = 1:size(M.pairs, 1)
    a = M.pairs(q, 1); b = M.pairs(q, 2);
    ia = Y == M.classes{a}; ib = Y == M.classes{b};
    Xq = [X(ia, :); X(ib, :)];
    yq = [ones(nnz(ia), 1); -ones(nnz(ib), 1)];
    Cq = C * ones(numel(yq), 1);
    Cq(yq > 0) = C * numel(yq) / (2 * nnz(ia));
    Cq(yq < 0) = C * numel(yq) / (2 * nnz(ib));
    [alpha, b0] = smo_solve(rbf(Xq, Xq, gamma), yq, Cq);
    sv = alpha > 1e-8;
    M.bin{q} = struct('SV', Xq(sv, :), 'coef', alpha(sv) .* yq(sv), 'b', b0);
end
end

function [alpha, b] = smo_solve(Kmat, y, Cv)
n = numel(y);
alpha = zeros(n, 1);
G = -ones(n, 1);
Kd = diag(Kmat);
tol = 1e-3; tau = 1e-12;
maxIter = max(1e5, 50 * n);
for it = 1:maxIter
    yG = -y .* G;
    up  = (y > 0 & alpha < Cv) | (y < 0 & alpha > 0);
    low = (y > 0 & alpha > 0)  | (y < 0 & alpha < Cv);
    yGu = yG; yGu(~up) = -inf;
    [m, i] = max(yGu);
    yGl = yG; yGl(~low) = inf;
    if m - min(yGl) < tol, break; end
    bij = m - yG;
    aij = Kd(i) + Kd - 2 * Kmat(:, i); aij(aij <= 0) = tau;
    obj = -(bij .^ 2) ./ aij;
    obj(~low | yG >= m) = inf;
    [~, j] = min(obj);
    ai = alpha(i); aj = alpha(j);
    a = max(Kd(i) + Kd(j) - 2 * Kmat(i, j), tau);
    bb = -y(i) * G(i) + y(j) * G(j);
    s = y(i) * ai + y(j) * aj;
    ni = ai + y(i) * bb / a;
    ni = min(max(ni, 0), Cv(i));
    nj = y(j) * (s - y(i) * ni);
    nj = min(max(nj, 0), Cv(j));
    ni = y(i) * (s - y(j) * nj);
    ni = min(max(ni, 0), Cv(i));
    dai = ni - ai; daj = nj - aj;
    alpha(i) = ni; alpha(j) = nj;
    G = G + y .* (Kmat(:, i) * (y(i) * dai) + Kmat(:, j) * (y(j) * daj));
end
yG = -y .* G;
free = alpha > 1e-8 & alpha < Cv - 1e-8;
if any(free)
    b = mean(yG(free));
else
    up  = (y > 0 & alpha < Cv) | (y < 0 & alpha > 0);
    low = (y > 0 & alpha > 0)  | (y < 0 & alpha < Cv);
    b = (max(yG(up)) + min(yG(low))) / 2;
end
end

function Kmat = rbf(A, B, gamma)
d = sum(A.^2, 2) + sum(B.^2, 2).' - 2 * (A * B.');
Kmat = exp(-gamma * max(d, 0));
end
