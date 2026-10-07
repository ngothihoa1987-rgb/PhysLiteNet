function D = prepare_dataset(dataset, cfg)
dataset = upper(string(dataset));
cacheFile = fullfile(cfg.paths.cache, dataset + ".mat");
if isfile(cacheFile) && ~cfg.forceRebuild
    fprintf('[%s] Loading cache %s\n', dataset, cacheFile);
    D = load(cacheFile);
    return
end
if ~isfolder(cfg.paths.cache), mkdir(cfg.paths.cache); end

switch dataset
    case "PU",   T = pu_file_list(cfg);
    case "HUST", T = hust_file_list(cfg);
    case "XJTU", T = xjtu_file_list(cfg);
end
if ~ismember('split', T.Properties.VariableNames), T.split = repmat("test", height(T), 1); end
fprintf('[%s] %d files. Extracting...\n', dataset, height(T));

o = bearing_orders(dataset);
baseOrders = arrayfun(@(s) o.(s), cfg.phys.baseOrders);
[sos, g] = design_bandpass(cfg);

out = cell(height(T), 1);
t0 = tic;
if cfg.useParallel
    parfor i = 1:height(T)
        out{i} = process_one_file(dataset, T(i,:), baseOrders, sos, g, cfg);
    end
else
    for i = 1:height(T)
        out{i} = process_one_file(dataset, T(i,:), baseOrders, sos, g, cfg);
        if mod(i, 50) == 0 || i == height(T)
            fprintf('  %d/%d file (%.0f s)\n', i, height(T), toc(t0));
        end
    end
end

nP = cellfun(@(s) size(s.P, 3), out);
nR = cellfun(@(s) size(s.R, 2), out);
L = cfg.phys.L; C = numel(baseOrders); W = cfg.raw.winLen;
D.Xphys = zeros(1, L, C, sum(nP), 'single');
D.Xraw  = zeros(1, W, 1, sum(nR), 'single');
D.feat  = zeros(sum(nP), numel(out{1}.F(1,:)), 'single');
iP = 0; iR = 0;
fileIdxP = zeros(sum(nP),1); fileIdxR = zeros(sum(nR),1);
T.speedHz = cellfun(@(s) s.fr, out);
for i = 1:height(T)
    k = nP(i); if k > 0
        D.Xphys(1,:,:,iP+(1:k)) = reshape(out{i}.P, 1, L, C, k);
        D.feat(iP+(1:k), :) = out{i}.F;
        fileIdxP(iP+(1:k)) = i; iP = iP + k;
    end
    k = nR(i); if k > 0
        D.Xraw(1,:,1,iR+(1:k)) = reshape(out{i}.R, 1, W, 1, k);
        fileIdxR(iR+(1:k)) = i; iR = iR + k;
    end
end
cats = cellstr(cfg.classes);
D.Yphys = categorical(T.label(fileIdxP), cats);
D.Yraw  = categorical(T.label(fileIdxR), cats);
D.metaPhys = T(fileIdxP, :); D.metaPhys.fileId = fileIdxP;
D.metaRaw  = T(fileIdxR, :); D.metaRaw.fileId  = fileIdxR;
[~, D.featNames] = envelope_features(out{1}.P(:,:,1), randn(100,1), abs(randn(100,1)), cfg);
D.fileTable = T;
D.dataset = dataset;
D.baseOrders = baseOrders;
fprintf('[%s] Xphys: %d samples, Xraw: %d samples. Saving cache...\n', dataset, size(D.Xphys,4), size(D.Xraw,4));
save(cacheFile, '-struct', 'D', '-v7.3');
end

function s = process_one_file(dataset, row, baseOrders, sos, g, cfg)
fs = cfg.fs;
[x, fr] = load_signal(dataset, row, cfg);
if cfg.speed.refine && dataset ~= "PU"
    fr = refine_shaft_freq(x, fs, fr, cfg.speed.relRange.(char(dataset)));
end
xb  = filtfilt(sos, g, x);
env = abs(hilbert(xb));

segLen = round(cfg.phys.nRev * fs / fr);
if numel(x) < segLen, segLen = numel(x); end
hop = max(1, round(segLen * (1 - cfg.phys.overlap)));
st = 1:hop:(numel(x) - segLen + 1);
if numel(st) > cfg.phys.maxSegPerFile
    st = st(unique(round(linspace(1, numel(st), cfg.phys.maxSegPerFile))));
end
P = zeros(cfg.phys.L, numel(baseOrders), numel(st), 'single');
F = [];
for k = 1:numel(st)
    idx = st(k):st(k) + segLen - 1;
    P(:,:,k) = envelope_order_spectrum(env(idx), fs, fr, baseOrders, cfg);
    F(k,:) = envelope_features(P(:,:,k), xb(idx), env(idx), cfg);
end

W = cfg.raw.winLen;
nW = cfg.raw.winPerFile.(char(dataset));
stR = unique(round(linspace(1, numel(x) - W + 1, nW)));
R = zeros(W, numel(stR), 'single');
for k = 1:numel(stR)
    v = x(stR(k):stR(k) + W - 1);
    R(:,k) = single((v - mean(v)) / (std(v) + eps));
end
s.P = P; s.F = single(F); s.R = R; s.fr = fr;
end
