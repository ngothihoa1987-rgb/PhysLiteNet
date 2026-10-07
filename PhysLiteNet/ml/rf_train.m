function F = rf_train(X, Y, nTrees, minLeaf)
if nargin < 3 || isempty(nTrees), nTrees = 200; end
if nargin < 4 || isempty(minLeaf), minLeaf = 1; end
X = double(X);
F.classes = categories(Y);
yi = double(Y);
K = numel(F.classes);
[N, p] = size(X);
mtry = max(1, floor(sqrt(p)));
idxC = arrayfun(@(k) find(yi == k), 1:K, 'UniformOutput', false);
nPer = round(N / K);
F.trees = cell(nTrees, 1);
for t = 1:nTrees
    boot = cell2mat(cellfun(@(ic) ic(randi(numel(ic), nPer, 1)), idxC(:), 'UniformOutput', false));
    F.trees{t} = grow_tree(X(boot, :), yi(boot), K, mtry, minLeaf);
end
end

function T = grow_tree(X, y, K, mtry, minLeaf)
N = numel(y);
cap = 2 * N;
T.feat = zeros(cap, 1); T.thr = zeros(cap, 1);
T.left = zeros(cap, 1); T.right = zeros(cap, 1);
T.prob = zeros(cap, K);
stackNode = 1; stackIdx = {(1:N).'};
nNodes = 1;
while ~isempty(stackNode)
    nd = stackNode(end); idx = stackIdx{end};
    stackNode(end) = []; stackIdx(end) = [];
    yy = y(idx);
    cnt = accumarray(yy, 1, [K 1]).';
    T.prob(nd, :) = cnt / sum(cnt);
    if numel(idx) < 2 * minLeaf || nnz(cnt) <= 1, continue; end
    [f, thr] = best_split(X(idx, :), yy, K, mtry, minLeaf);
    if f == 0, continue; end
    goL = X(idx, f) <= thr;
    T.feat(nd) = f; T.thr(nd) = thr;
    T.left(nd) = nNodes + 1; T.right(nd) = nNodes + 2;
    stackNode(end+1:end+2) = [nNodes + 1, nNodes + 2];
    stackIdx(end+1:end+2) = {idx(goL), idx(~goL)};
    nNodes = nNodes + 2;
end
T.feat = T.feat(1:nNodes); T.thr = T.thr(1:nNodes);
T.left = T.left(1:nNodes); T.right = T.right(1:nNodes); T.prob = T.prob(1:nNodes, :);
end

function [bf, bt] = best_split(X, y, K, mtry, minLeaf)
n = numel(y);
feats = randperm(size(X, 2), mtry);
best = inf; bf = 0; bt = 0;
Y1 = full(sparse(1:n, y, 1, n, K));
tot = sum(Y1, 1);
for f = feats
    [xs, o] = sort(X(:, f));
    cl = cumsum(Y1(o, :), 1);
    nl = (1:n).'; nr = n - nl;
    cr = tot - cl;
    gl = 1 - sum(cl.^2, 2) ./ nl.^2;
    gr = 1 - sum(cr.^2, 2) ./ max(nr, 1).^2;
    imp = (nl .* gl + nr .* gr) / n;
    ok = [xs(1:end-1) < xs(2:end); false] & nl >= minLeaf & nr >= minLeaf;
    imp(~ok) = inf;
    [v, k] = min(imp);
    if v < best
        best = v; bf = f; bt = (xs(k) + xs(k+1)) / 2;
    end
end
if ~isfinite(best), bf = 0; end
end
