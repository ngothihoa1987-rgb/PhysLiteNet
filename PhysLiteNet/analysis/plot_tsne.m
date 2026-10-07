function plot_tsne(cfg, DS, model, inputType, seed)
if nargin < 3, model = "PhysLiteNet"; end
if nargin < 4, inputType = "physics"; end
if nargin < 5, seed = 1; end
S = load(fullfile(cfg.paths.results, 'runs', sprintf('%s__%s__s%d.mat', model, inputType, seed)), 'net');
layer = 'gap';
if any(strcmp({S.net.Layers.Name}, 'cat')), layer = 'cat'; end
names = ["PU" DS.targets]; mk = 'osd^';
E = []; Y = []; G = [];
rng(0);
for i = 1:numel(names)
    [X, y] = get_split(DS.(names(i)), inputType, "all");
    if names(i) == "PU"
        [X, y] = get_split(DS.PU, inputType, "val");
    end
    k = randperm(numel(y), min(800, numel(y)));
    e = minibatchpredict(S.net, X(:, :, :, k), 'Outputs', layer, 'InputDataFormats', 'SSCB');
    e = squeeze(gather(e)).';
    E = [E; e]; Y = [Y; y(k)]; G = [G; repmat(i, numel(k), 1)];
end
if has_stats_toolbox()
    Z = tsne(double(E), 'Perplexity', 30);
else
    Z = tsne_simple(E, 30, 1000);
end
f = figure('Color', 'w', 'Position', [100 100 620 500]); hold on
col = lines(numel(cfg.classes));
dsName = containers.Map({'PU','HUST','XJTU'}, {'PU (val.)','HUST','XJTU-SY'});
clsName = ["Healthy" "Outer race" "Inner race"];
inName = containers.Map({'physics','physref','raw'}, {'aligned input','healthy reference','raw input'});
for i = 1:numel(names)
    for c = 1:numel(cfg.classes)
        m = G == i & Y == cfg.classes(c);
        scatter(Z(m,1), Z(m,2), 12, col(c,:), mk(i), 'filled', 'MarkerFaceAlpha', 0.6, ...
            'DisplayName', string(dsName(char(names(i)))) + ", " + clsName(c));
    end
end
legend('Location', 'bestoutside'); axis off
mdl = strrep(strrep(char(model), '_1D', '-1D'), '1D18', '1D-18');
title(sprintf('%s, %s', mdl, inName(char(inputType))), 'Interpreter', 'none');
exportgraphics(f, fullfile(cfg.paths.results, 'figures', sprintf('tsne_%s_%s.png', model, inputType)), 'Resolution', 300);
close(f);
end
