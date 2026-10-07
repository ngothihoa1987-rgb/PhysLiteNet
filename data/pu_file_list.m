function T = pu_file_list(cfg)
files = dir(fullfile(cfg.paths.PU, '**', '*.mat'));
if isempty(files), error('No .mat files found in %s', cfg.paths.PU); end

n = numel(files);
path = strings(n,1); bearing = strings(n,1); cond = strings(n,1);
for i = 1:n
    path(i) = string(fullfile(files(i).folder, files(i).name));
    tok = regexp(files(i).name, '^(N\d+_M\d+_F\d+)_(K\w\d+)_\d+\.mat$', 'tokens', 'once');
    if isempty(tok), continue; end
    cond(i) = tok{1}; bearing(i) = tok{2};
end
keep = bearing ~= "";
path = path(keep); bearing = bearing(keep); cond = cond(keep);

label = strings(numel(path),1);
label(ismember(bearing, cfg.pu.healthy)) = "Healthy";
label(ismember(bearing, cfg.pu.outer))   = "OuterRace";
label(ismember(bearing, cfg.pu.inner))   = "InnerRace";
use = label ~= "";

split = repmat("train", numel(path), 1);
split(ismember(bearing, cfg.pu.valBearings)) = "val";
if isfield(cfg.pu, 'testBearings')
    split(ismember(bearing, cfg.pu.testBearings)) = "test";
end

origin = repmat("real", numel(path), 1);
origin(ismember(bearing, cfg.pu.artificial)) = "artificial";
origin(label == "Healthy") = "healthy";

T = table(path(use), bearing(use), cond(use), label(use), split(use), origin(use), ...
    'VariableNames', {'path','bearing','condition','label','split','origin'});
T.speedHz = nan(height(T),1);

if cfg.debug
    [~, ia] = unique(T.bearing + T.condition, 'stable');
    idx = sort([ia; ia+1]);
    T = T(idx(idx <= height(T)), :);
end
end
