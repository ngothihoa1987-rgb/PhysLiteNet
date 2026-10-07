function S = rf_predict(F, X)
X = double(X);
N = size(X, 1);
S = zeros(N, numel(F.classes));
for t = 1:numel(F.trees)
    T = F.trees{t};
    nd = ones(N, 1);
    active = T.left(nd) > 0;
    while any(active)
        a = find(active);
        goL = X(sub2ind(size(X), a, T.feat(nd(a)))) <= T.thr(nd(a));
        nd(a(goL))  = T.left(nd(a(goL)));
        nd(a(~goL)) = T.right(nd(a(~goL)));
        active = T.left(nd) > 0;
    end
    S = S + T.prob(nd, :);
end
S = S / numel(F.trees);
end
