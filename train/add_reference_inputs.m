function D = add_reference_inputs(D, cfg)
M = D.metaPhys;
S = expm1(double(D.Xphys));
[~, L, C, N] = size(S);
Sref = nan(1, L, C, N);
isH = D.Yphys == "Healthy";
switch string(D.dataset)
    case "PU"
        conds = unique(M.condition);
        for c = conds.'
            inC = M.condition == c;
            hb = unique(M.bearing(inC & isH & M.split == "train"));
            Mb = zeros(1, L, C, numel(hb));
            for j = 1:numel(hb)
                Mb(:, :, :, j) = median(S(:, :, :, inC & M.bearing == hb(j)), 4);
            end
            refAll = median(Mb, 4);
            idx = find(inC);
            for i = idx.'
                j = find(hb == M.bearing(i));
                if isempty(j)
                    Sref(:, :, :, i) = refAll;
                else
                    Sref(:, :, :, i) = median(Mb(:, :, :, setdiff(1:numel(hb), j)), 4);
                end
            end
        end
    case "HUST"
        sp = unique(M.condition);
        Ms = nan(1, L, C, numel(sp));
        for j = 1:numel(sp)
            k = isH & M.condition == sp(j);
            if any(k), Ms(:, :, :, j) = median(S(:, :, :, k), 4); end
        end
        for j = 1:numel(sp)
            other = setdiff(1:numel(sp), j);
            ref = median(Ms(:, :, :, other), 4, 'omitnan');
            inJ = M.condition == sp(j);
            Sref(:, :, :, inJ) = repmat(ref, 1, 1, 1, nnz(inJ));
        end
    case "XJTU"
        bs = unique(M.bearing);
        for b = bs.'
            inB = M.bearing == b;
            k = inB & M.split == "ref";
            if ~any(k)
                h = find(inB & isH);
                [~, o] = sort(M.fileIndex(h));
                k = false(N, 1); k(h(o(1:min(3, numel(o))))) = true;
            end
            ref = median(S(:, :, :, k), 4);
            Sref(:, :, :, inB) = repmat(ref, 1, 1, 1, nnz(inB));
        end
end
miss = squeeze(any(isnan(Sref), [1 2 3]));
if any(miss)
    warning('%s: %d samples without reference spectrum, using a flat reference.', D.dataset, nnz(miss));
    Sref(:, :, :, miss) = 1;
end
R = S ./ max(Sref, cfg.ref.floor);
R = R ./ (median(R, 2) + eps);
D.XphysRef = single(log1p(R));
Fh = harmonic_features(D.XphysRef, cfg);
D.featRef = [Fh, D.feat(:, size(Fh, 2) + 1:end)];
end
