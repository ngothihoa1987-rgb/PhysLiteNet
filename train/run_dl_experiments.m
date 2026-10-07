function run_dl_experiments(cfg, DS, models, inputs, seeds)
if nargin < 3, models = cfg.models; end
if nargin < 4, inputs = cfg.inputs; end
if nargin < 5, seeds  = cfg.seeds;  end
runDir = fullfile(cfg.paths.results, 'runs');
if ~isfolder(runDir), mkdir(runDir); end

for inp = inputs
    [XTr, YTr] = get_split(DS.PU, inp, "train");
    [XVa, YVa, mVa] = get_split(DS.PU, inp, "val");
    [XTe, YTe, mTe] = get_split(DS.PU, inp, "test");
    inputSize = size(XTr, 1:3);
    for mdl = models
        useAug = cfg.aug.enable && any(inp == ["physics" "physref"]) && ~endsWith(mdl, "_noAug");
        augFn = [];
        if useAug, augFn = @(X) physics_augment(X, cfg, DS.PU.baseOrders); end
        for seed = seeds
            f = fullfile(runDir, sprintf('%s__%s__s%d.mat', mdl, inp, seed));
            if isfile(f), fprintf('Found %s, skipping\n', f); continue; end
            fprintf('\n===== %s | %s | seed %d | aug=%d =====\n', mdl, inp, seed, useAug);
            rng(seed, 'twister');
            try, gpurng(seed, 'Philox'); catch, end
            net = build_model(mdl, inputSize, numel(cfg.classes), cfg, inp);
            [net, info, tTrain] = train_dl_model(net, XTr, YTr, XVa, YVa, cfg, augFn);

            r = struct('model', mdl, 'input', inp, 'seed', seed, 'trainTime', tTrain, ...
                       'aug', useAug, 'round', cfg.round);
            r.params = sum(cellfun(@numel, net.Learnables.Value));
            r.history = info.ValidationHistory;
            r.bestEpoch = info.bestEpoch;
            r.PU_val  = evaluate_net(net, XVa, YVa, mVa, cfg);
            r.PU_test = evaluate_net(net, XTe, YTe, mTe, cfg);
            for t = DS.targets
                [Xt, Yt, mt] = get_split(DS.(t), inp, "all");
                r.(t) = evaluate_net(net, Xt, Yt, mt, cfg);
                fprintf('  %-7s acc=%.4f  macroF1=%.4f  (file acc=%.4f)\n', t, r.(t).acc, r.(t).f1, r.(t).accFile);
            end
            fprintf('  PU_val  acc=%.4f  macroF1=%.4f\n', r.PU_val.acc, r.PU_val.f1);
            fprintf('  PU_test acc=%.4f  macroF1=%.4f\n', r.PU_test.acc, r.PU_test.f1);
            save(f, 'r', 'net');
        end
    end
end
end

function m = evaluate_net(net, X, Y, meta, cfg)
scores = minibatchpredict(net, X, 'MiniBatchSize', 512, 'InputDataFormats', 'SSCB');
if isa(scores, 'dlarray'), scores = extractdata(scores); end
scores = gather(scores);
m = compute_metrics(Y, scores, cfg.classes, meta.fileId);
m.yTrue = Y;
m.scores = single(scores);
end
