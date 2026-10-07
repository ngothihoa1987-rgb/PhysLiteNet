function Y = tsne_simple(X, perplexity, nIter)
if nargin < 2, perplexity = 30; end
if nargin < 3, nIter = 1000; end
X = double(X);
X = X - mean(X, 1);
[~, ~, V] = svd(X, 'econ');
X = X * V(:, 1:min(50, size(V, 2)));
N = size(X, 1);
D = max(sum(X.^2, 2) + sum(X.^2, 2).' - 2 * (X * X.'), 0);
P = zeros(N);
logU = log(perplexity);
for i = 1:N
    di = D(i, [1:i-1, i+1:N]);
    beta = 1; lo = -inf; hi = inf;
    for t = 1:50
        pr = exp(-di * beta); sp = max(sum(pr), realmin);
        H = log(sp) + beta * sum(di .* pr) / sp;
        if abs(H - logU) < 1e-5, break; end
        if H > logU, lo = beta; if isinf(hi), beta = beta * 2; else, beta = (beta + hi) / 2; end
        else,        hi = beta; if isinf(lo), beta = beta / 2; else, beta = (beta + lo) / 2; end
        end
    end
    P(i, [1:i-1, i+1:N]) = pr / sp;
end
P = (P + P.') / (2 * N);
P = max(P, 1e-12);
Y = 1e-4 * randn(N, 2);
dY = zeros(N, 2); gains = ones(N, 2);
for it = 1:nIter
    mom = 0.5 + 0.3 * (it > 250);
    ex = 12 * (it <= 100) + (it > 100);
    num = 1 ./ (1 + max(sum(Y.^2, 2) + sum(Y.^2, 2).' - 2 * (Y * Y.'), 0));
    num(1:N+1:end) = 0;
    Q = max(num / sum(num(:)), 1e-12);
    L = (ex * P - Q) .* num;
    G = 4 * (diag(sum(L, 1)) - L) * Y;
    gains = (gains + 0.2) .* (sign(G) ~= sign(dY)) + (gains * 0.8) .* (sign(G) == sign(dY));
    gains = max(gains, 0.01);
    dY = mom * dY - 200 * (gains .* G);
    Y = Y + dY;
    Y = Y - mean(Y, 1);
end
end
