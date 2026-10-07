function S = analyze_results(cfg)
outDir = cfg.paths.results;
figDir = fullfile(outDir, 'figures'); if ~isfolder(figDir), mkdir(figDir); end
R = collect_results(cfg);
if isempty(R), warning('No results found.'); S = []; return; end
writetable(R, fullfile(outDir, 'all_runs.csv'));

metrics = ["acc" "bacc" "f1" "accFile" "f1File"];
S = groupsummary(R, {'setting','method','target'}, {'mean','std'}, metrics);
writetable(S, fullfile(outDir, 'summary_mean_std.csv'));
disp(S(:, {'setting','method','target','GroupCount','mean_f1','std_f1'}));

C = complexity_table(cfg);
if ~isempty(C), writetable(C, fullfile(outDir, 'complexity.csv')); disp(C); end

fid = fopen(fullfile(outDir, 'stat_tests.txt'), 'w');
settings = ["zero-shot" "healthy-reference"];
refs = ["PhysLiteNet (physics)" "PhysLiteNet (physref)"];
for g = 1:2
    Rg = R(R.setting == settings(g) & any(R.target == ["HUST" "XJTU"], 2), :);
    if isempty(Rg), continue; end
    fprintf(fid, '==================== %s ====================\n', upper(settings(g)));
    write_stats(fid, Rg, refs(g));
    write_latex_table(S(S.setting == settings(g), :), fullfile(outDir, "table_" + strrep(settings(g), '-', '_') + ".tex"));
end
fclose(fid);
type(fullfile(outDir, 'stat_tests.txt'));

targets = ["PU_val" "PU_test" "HUST" "XJTU"];
targets = targets(ismember(targets, unique(R.target)));
f = figure('Color', 'w', 'Position', [50 50 380 * numel(targets) 620]);
for k = 1:numel(targets)
    subplot(1, numel(targets), k);
    Sk = S(S.target == targets(k), :);
    [~, o] = sort(Sk.mean_f1, 'descend'); Sk = Sk(o, :);
    barh(Sk.mean_f1); hold on
    errorbar(Sk.mean_f1, 1:height(Sk), Sk.std_f1, 'horizontal', 'k.', 'LineWidth', 1);
    set(gca, 'YTick', 1:height(Sk), 'YTickLabel', Sk.method, 'YDir', 'reverse', ...
        'TickLabelInterpreter', 'none', 'FontSize', 7);
    xlim([0 1]); xlabel('Macro-F1'); title(strrep(targets(k), '_', ' ')); grid on
end
exportgraphics(f, fullfile(figDir, 'f1_by_method.png'), 'Resolution', 300); close(f);

if ~isempty(C)
    f = figure('Color', 'w', 'Position', [100 100 680 480]); hold on
    for i = 1:height(C)
        v = R.f1(R.method == C.method(i) & any(R.target == ["HUST" "XJTU"], 2));
        if isempty(v), continue; end
        mk = 'o'; if endsWith(C.method(i), "(physref)"), mk = 's'; end
        scatter(C.params(i), mean(v), 70, mk, 'filled');
        text(C.params(i) * 1.15, mean(v), C.method(i), 'Interpreter', 'none', 'FontSize', 7);
    end
    set(gca, 'XScale', 'log'); xlabel('Number of parameters (log)'); ylabel('Mean target macro-F1');
    grid on; box on
    exportgraphics(f, fullfile(figDir, 'f1_vs_params.png'), 'Resolution', 300); close(f);
end

for inp = ["physics" "physref"]
    for t = ["PU_test" "HUST" "XJTU"]
        cm = 0; ok = false;
        for s = cfg.seeds
            fr = fullfile(cfg.paths.results, 'runs', sprintf('PhysLiteNet__%s__s%d.mat', inp, s));
            if ~isfile(fr), continue; end
            L = load(fr, 'r');
            if isfield(L.r, t), cm = cm + L.r.(t).cm; ok = true; end
        end
        if ~ok, continue; end
        f = figure('Color', 'w', 'Position', [100 100 420 360]);
        confusionchart(cm, cellstr(cfg.classes), 'RowSummary', 'row-normalized', ...
            'Title', "PhysLiteNet (" + inp + ") - " + strrep(t, '_', ' '));
        exportgraphics(f, fullfile(figDir, "cm_PhysLiteNet_" + inp + "_" + t + ".png"), 'Resolution', 300);
        close(f);
    end
end
fprintf('Results saved in %s\n', outDir);
end

function write_stats(fid, Rg, ref)
Rg.block = Rg.target + "_" + string(Rg.seed);
blocks = unique(Rg.block);
meths = unique(Rg.method);
M = nan(numel(blocks), numel(meths));
for j = 1:numel(meths)
    for b = 1:numel(blocks)
        v = Rg.f1(Rg.method == meths(j) & Rg.block == blocks(b));
        if ~isempty(v), M(b, j) = v(1); end
    end
end
full = all(~isnan(M), 1) & ~contains(meths, "_no").';
if nnz(full) >= 2 && size(M, 1) >= 2
    [p, chi2, rk] = stat_friedman(M(:, full));
    fprintf(fid, 'Friedman (macro-F1, %d blocks, %d methods): chi2 = %.2f, p = %.3g\n', ...
        size(M, 1), nnz(full), chi2, p);
    fm = meths(full);
    [rks, o] = sort(rk);
    for j = 1:numel(o), fprintf(fid, '  mean rank %-38s %.2f\n', fm(o(j)), rks(j)); end
    if any(~full)
        fprintf(fid, '  (excluded from Friedman, missing blocks or ablation: %s)\n', strjoin(meths(~full), ', '));
    end
end
iRef = find(meths == ref);
if isempty(iRef), fprintf(fid, '\n(%s not found)\n\n', ref); return; end
others = setdiff(1:numel(meths), iRef);
pw = nan(size(others)); d = pw; nb = pw;
for k = 1:numel(others)
    c = ~isnan(M(:, iRef)) & ~isnan(M(:, others(k)));
    nb(k) = nnz(c);
    if nb(k) >= 2
        pw(k) = stat_signrank(M(c, iRef), M(c, others(k)));
        d(k) = mean(M(c, iRef) - M(c, others(k)));
    end
end
pH = pw; okp = ~isnan(pw); pH(okp) = holm(pw(okp));
fprintf(fid, '\nWilcoxon signed-rank: %s vs each method (Holm-corrected)\n', ref);
for k = 1:numel(others)
    fprintf(fid, '  vs %-38s dF1 = %+.4f  n = %2d  p = %.4g  p_Holm = %.4g\n', ...
        meths(others(k)), d(k), nb(k), pw(k), pH(k));
end
fprintf(fid, '\n');
end

function C = complexity_table(cfg)
files = dir(fullfile(cfg.paths.results, 'runs', '*__s1.mat'));
rows = {};
for i = 1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name));
    if ~isfield(S, 'net'), continue; end
    inSz = S.net.Layers(1).InputSize;
    c = model_complexity(S.net, inSz);
    rows(end+1,:) = {S.r.model + " (" + S.r.input + ")", c.params, c.MMACs, c.latencyMs, c.sizeMB};
end
if isempty(rows), C = table(); return; end
C = cell2table(rows, 'VariableNames', {'method','params','MMACs','latencyMs_CPU','sizeMB'});
end

function p = holm(p)
[ps, o] = sort(p); m = numel(p);
adj = min(1, cummax((m - (1:m) + 1) .* ps));
p(o) = adj;
end

function write_latex_table(S, file)
if isempty(S), return; end
fid = fopen(file, 'w');
targets = ["PU_val" "PU_test" "HUST" "XJTU"];
targets = targets(ismember(targets, unique(S.target)));
meths = unique(S.method, 'stable');
fprintf(fid, '%% Macro-F1 (%%), mean $\\pm$ std\n\\begin{tabular}{l%s}\n\\hline\nMethod', repmat('c', 1, numel(targets) + 1));
fprintf(fid, ' & %s', strrep(targets, '_', '-')); fprintf(fid, ' & Mean (targets) \\\\\n\\hline\n');
for j = 1:numel(meths)
    fprintf(fid, '%s', strrep(meths(j), '_', '\_'));
    mt = [];
    for t = targets
        k = S.method == meths(j) & S.target == t;
        if any(k)
            fprintf(fid, ' & %.2f $\\pm$ %.2f', 100 * S.mean_f1(k), 100 * S.std_f1(k));
            if any(t == ["HUST" "XJTU"]), mt(end+1) = S.mean_f1(k); end
        else
            fprintf(fid, ' & --');
        end
    end
    if numel(mt) == 2, fprintf(fid, ' & %.2f', 100 * mean(mt)); else, fprintf(fid, ' & --'); end
    fprintf(fid, ' \\\\\n');
end
fprintf(fid, '\\hline\n\\end{tabular}\n');
fclose(fid);
end
