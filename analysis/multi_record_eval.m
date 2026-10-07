function T = multi_record_eval(cfg, DS)
files = dir(fullfile(cfg.paths.results, 'runs', '*.mat'));
rows = {};
for i = 1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name), 'r');
    r = S.r;
    for t = ["PU_test" "HUST" "XJTU"]
        if ~isfield(r, t) || ~isfield(r.(t), 'scores'), continue; end
        if t == "PU_test"
            [~, ~, meta] = get_split(DS.PU, string(r.input), "test");
        else
            [~, ~, meta] = get_split(DS.(t), string(r.input), "all");
        end
        if height(meta) ~= size(r.(t).scores, 1), continue; end
        for W = cfg.multi.windows
            [Sw, Yw] = window_scores(r.(t).scores, r.(t).yTrue, meta, W);
            m = compute_metrics(Yw, Sw, cfg.classes, []);
            rows(end+1, :) = {string(r.model) + " (" + string(r.input) + ")", r.seed, t, W, m.acc, m.f1, numel(Yw)};
        end
    end
end
if isempty(rows), T = table(); warning('No scores available for multi-record evaluation.'); return; end
T = cell2table(rows, 'VariableNames', {'method','seed','target','W','acc','f1','nWindows'});
G = groupsummary(T, {'method','target','W'}, {'mean','std'}, {'acc','f1'});
writetable(G, fullfile(cfg.paths.results, 'multi_record.csv'));
disp(G(:, {'method','target','W','mean_f1','std_f1'}));
end

function [Sw, Yw] = window_scores(S, Y, meta, W)
key = string(meta.bearing) + "|" + string(meta.condition) + "|" + string(Y);
if ismember('fileIndex', meta.Properties.VariableNames)
    ord = double(meta.fileIndex);
else
    ord = double(meta.fileId);
end
[g, ~] = findgroups(key);
Sw = zeros(0, size(S, 2)); Yw = Y([]);
for k = 1:max(g)
    idx = find(g == k);
    [~, o] = sortrows([ord(idx), (1:numel(idx)).']);
    idx = idx(o);
    nWin = floor(numel(idx) / W);
    for w = 1:nWin
        ii = idx((w - 1) * W + (1:W));
        Sw(end+1, :) = mean(S(ii, :), 1);
        Yw(end+1, 1) = Y(ii(1));
    end
end
end
