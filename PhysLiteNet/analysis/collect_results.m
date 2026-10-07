function R = collect_results(cfg)
files = dir(fullfile(cfg.paths.results, 'runs', '*.mat'));
rows = {};
for i = 1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name), 'r');
    r = S.r;
    for t = ["PU_val" "PU_test" "HUST" "XJTU"]
        if ~isfield(r, t), continue; end
        m = r.(t);
        rows(end+1,:) = {string(r.model), string(r.input), r.seed, t, m.acc, m.bacc, m.f1, ...
            m.accFile, m.f1File, r.params, r.trainTime};
    end
end
if isempty(rows), R = table(); return; end
R = cell2table(rows, 'VariableNames', {'model','input','seed','target','acc','bacc','f1', ...
    'accFile','f1File','params','trainTime'});
R.method = R.model + " (" + R.input + ")";
R.setting = repmat("zero-shot", height(R), 1);
R.setting(endsWith(R.input, "ref")) = "healthy-reference";
end
