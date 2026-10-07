function run_ml_baselines(cfg, DS, seeds)
if nargin < 3, seeds = cfg.seeds; end
runDir = fullfile(cfg.paths.results, 'runs');
if ~isfolder(runDir), mkdir(runDir); end

for inp = cfg.ml.inputs
    [Ftr, Ytr] = get_split(DS.PU, inp, "train");
    [Fva, Yva, mVa] = get_split(DS.PU, inp, "val");
    [Fte, Yte, mTe] = get_split(DS.PU, inp, "test");
    Ftr = double(Ftr);
    for mdl = ["SVM" "RF"]
        for seed = seeds
            f = fullfile(runDir, sprintf('%s__%s__s%d.mat', mdl, inp, seed));
            if isfile(f), fprintf('Found %s, skipping\n', f); continue; end
            rng(seed, 'twister');
            t0 = tic;
            switch mdl
                case "SVM"
                    k = randperm(numel(Ytr), round(0.8 * numel(Ytr)));
                    M = svm_ovo_train(Ftr(k, :), Ytr(k), cfg.ml.svmC, []);
                    predictFn = @(F) svm_ovo_predict(M, F);
                case "RF"
                    M = rf_train(Ftr, Ytr, cfg.ml.rfTrees, 1);
                    predictFn = @(F) rf_predict(M, F);
            end
            r = struct('model', mdl, 'input', inp, 'seed', seed, 'trainTime', toc(t0), ...
                       'aug', false, 'round', cfg.round);
            r.params = NaN;
            r.PU_val  = eval_ml(predictFn, Fva, Yva, mVa, cfg);
            r.PU_test = eval_ml(predictFn, Fte, Yte, mTe, cfg);
            fprintf('%s %s seed %d | PU_val F1=%.4f | PU_test F1=%.4f (%.0f s)\n', mdl, inp, seed, ...
                r.PU_val.f1, r.PU_test.f1, r.trainTime);
            for tg = DS.targets
                [Ft, Yt, mt] = get_split(DS.(tg), inp, "all");
                r.(tg) = eval_ml(predictFn, Ft, Yt, mt, cfg);
                fprintf('%s %s seed %d | %-5s acc=%.4f macroF1=%.4f\n', mdl, inp, seed, tg, r.(tg).acc, r.(tg).f1);
            end
            model = M;
            save(f, 'r', 'model');
        end
    end
end
end

function m = eval_ml(predictFn, F, Y, meta, cfg)
S = predictFn(double(F));
m = compute_metrics(Y, S, cfg.classes, meta.fileId);
m.yTrue = Y;
m.scores = single(S);
end
