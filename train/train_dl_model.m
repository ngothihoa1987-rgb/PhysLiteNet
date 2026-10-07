function [net, info, tTrain] = train_dl_model(net, XTr, YTr, XVa, YVa, cfg, augFn)
if nargin < 7, augFn = []; end
classes = categories(YTr);
K = numel(classes);
idxAll = oversample_idx(YTr);
N = numel(idxAll);
mb = min(cfg.train.miniBatch, N);
itPerEpoch = max(1, floor(N / mb));
totalIt = cfg.train.maxEpochs * itPerEpoch;
useGPU = ~strcmp(cfg.train.execEnv, 'cpu') && canUseGPU;
yTrIdx = double(YTr);

avgG = []; sqG = [];
it = 0;
best.f1 = -inf; best.loss = inf; best.net = net; best.epoch = 0;
wait = 0;
hist = zeros(0, 5);
t0 = tic;
for epoch = 1:cfg.train.maxEpochs
    perm = idxAll(randperm(N));
    trLoss = 0;
    for i = 1:itPerEpoch
        it = it + 1;
        bi = perm((i - 1) * mb + (1:mb));
        Xb = XTr(:, :, :, bi);
        if ~isempty(augFn), Xb = augFn(Xb); end
        Xb = dlarray(single(Xb), 'SSCB');
        Tb = zeros(K, mb, 'single');
        Tb(sub2ind([K mb], yTrIdx(bi).', 1:mb)) = 1;
        Tb = dlarray(Tb, 'CB');
        if useGPU, Xb = gpuArray(Xb); Tb = gpuArray(Tb); end
        [loss, grads, state] = dlfeval(@model_loss, net, Xb, Tb);
        net.State = state;
        grads = dlupdate(@(g, w) g + cfg.train.l2 * w, grads, net.Learnables);
        lr = 0.5 * cfg.train.learnRate * (1 + cos(pi * (it - 1) / totalIt));
        [net, avgG, sqG] = adamupdate(net, grads, avgG, sqG, it, lr);
        trLoss = trLoss + double(gather(extractdata(loss)));
    end
    S = predict_scores(net, XVa, useGPU);
    m = compute_metrics(YVa, S, classes, []);
    yv = double(YVa);
    vLoss = -mean(log(max(S(sub2ind(size(S), (1:numel(yv)).', yv)), 1e-12)));
    hist(end+1, :) = [epoch, trLoss / itPerEpoch, vLoss, m.acc, m.f1];
    if cfg.train.verbose
        fprintf('  epoch %2d | train loss %.4f | val loss %.4f | val acc %.4f | val F1 %.4f\n', hist(end, :));
    end
    if m.f1 > best.f1 + 1e-4 || (abs(m.f1 - best.f1) <= 1e-4 && vLoss < best.loss)
        best.f1 = m.f1; best.loss = vLoss; best.net = net; best.epoch = epoch; wait = 0;
    else
        wait = wait + 1;
        if wait >= cfg.train.patience, break; end
    end
end
tTrain = toc(t0);
net = best.net;
info.ValidationHistory = array2table(hist, 'VariableNames', {'Epoch','TrainLoss','ValLoss','ValAcc','ValF1'});
info.bestEpoch = best.epoch;
if cfg.train.verbose
    fprintf('  -> selected epoch %d (val F1 %.4f), %.0f s\n', best.epoch, best.f1, tTrain);
end
end

function [loss, grads, state] = model_loss(net, X, T)
[Y, state] = forward(net, X);
loss = crossentropy(Y, T);
grads = dlgradient(loss, net.Learnables);
end

function S = predict_scores(net, X, useGPU)
N = size(X, 4); bs = 512;
S = zeros(N, 0);
for s = 1:bs:N
    idx = s:min(N, s + bs - 1);
    Xb = dlarray(single(X(:, :, :, idx)), 'SSCB');
    if useGPU, Xb = gpuArray(Xb); end
    Yb = gather(extractdata(predict(net, Xb)));
    S(idx, 1:size(Yb, 1)) = Yb.';
end
end

function idx = oversample_idx(Y)
cats = categories(Y);
cnt = countcats(Y);
nMax = max(cnt);
idx = (1:numel(Y)).';
for c = 1:numel(cats)
    ic = find(Y == cats{c});
    if isempty(ic), continue; end
    idx = [idx; ic(randi(numel(ic), nMax - numel(ic), 1))];
end
end
