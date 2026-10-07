function [T, HI] = xjtu_file_list(cfg)

faultMap = containers.Map( ...
    {'1_1','1_2','1_3','1_4','1_5','2_1','2_2','2_3','2_4','2_5','3_1','3_2','3_3','3_4','3_5'}, ...
    {'OuterRace','OuterRace','OuterRace','Cage','Combined','InnerRace','OuterRace','Cage', ...
     'OuterRace','OuterRace','OuterRace','Combined','InnerRace','InnerRace','OuterRace'});

dirs = dir(fullfile(cfg.paths.XJTU, '**', 'Bearing*_*'));
dirs = dirs([dirs.isdir]);
if isempty(dirs)
    error(['No Bearing*_* folders found in %s.\n' ...
           'Extract the XJTU-SY archive first.'], cfg.paths.XJTU);
end

hiFile = fullfile(cfg.paths.cache, 'xjtu_health_index.mat');
if isfile(hiFile) && ~cfg.forceRebuild
    S = load(hiFile); HI = S.HI;
else
    HI = struct('bearing', {}, 'rms', {}, 'files', {}, 'fpt', {});
end

rows = {};
for b = 1:numel(dirs)
    tok = regexp(dirs(b).name, '^Bearing(\d)_(\d)$', 'tokens', 'once');
    if isempty(tok), continue; end
    id = [tok{1} '_' tok{2}];
    condIdx = str2double(tok{1});
    fdir = fullfile(dirs(b).folder, dirs(b).name);
    f = dir(fullfile(fdir, '*.csv'));
    num = cellfun(@(s) str2double(erase(s, '.csv')), {f.name});
    [num, o] = sort(num); f = f(o);
    paths = string(fullfile(fdir, {f.name}'));

    k = find(strcmp({HI.bearing}, id), 1);
    if isempty(k) || numel(HI(k).rms) ~= numel(paths)
        fprintf('  XJTU %s: computing RMS of %d files...\n', id, numel(paths));
        r = zeros(numel(paths),1);
        for i = 1:numel(paths)
            x = load_xjtu_signal(paths(i), cfg);
            r(i) = rms(x);
        end
        k = numel(HI) + 1;
        HI(k).bearing = id; HI(k).rms = r; HI(k).files = paths;
    end
    r = HI(k).rms; N = numel(r);
    nb = max(cfg.xjtu.minBaseline, round(cfg.xjtu.baselineFrac * N));
    thr = mean(r(1:nb)) + cfg.xjtu.kSigma * std(r(1:nb));
    above = r > thr; above(1:nb) = false;
    run = movsum(double(above), [0 cfg.xjtu.nConsec-1]) >= cfg.xjtu.nConsec;
    fpt = find(run, 1);
    if isempty(fpt), fpt = N + 1; end
    HI(k).fpt = fpt;

    ftype = faultMap(id);
    nRef = 0;
    if isfield(cfg, 'ref'), nRef = min(cfg.ref.xjtuFirstN, max(0, fpt - 2)); end
    for i = 1:nRef
        rows(end+1,:) = {paths(i), string(id), condIdx, cfg.xjtu.speedHz(condIdx), "Healthy", num(i), fpt, string(ftype), "ref"};
    end
    idxH = (nRef + 1):floor(cfg.xjtu.healthyFrac * (fpt - 1));
    idxH = pick_even(idxH, cfg.xjtu.maxHealthyFilesPerBearing);
    for i = idxH
        rows(end+1,:) = {paths(i), string(id), condIdx, cfg.xjtu.speedHz(condIdx), "Healthy", num(i), fpt, string(ftype), "test"};
    end
    if any(strcmp(ftype, {'OuterRace','InnerRace'})) && fpt <= N
        idxF = pick_even(fpt:N, cfg.xjtu.maxFaultFilesPerBearing);
        for i = idxF
            rows(end+1,:) = {paths(i), string(id), condIdx, cfg.xjtu.speedHz(condIdx), string(ftype), num(i), fpt, string(ftype), "test"};
        end
    end
    fprintf('  XJTU %s (%s): N=%d, FPT=%d, healthy=%d, fault=%d\n', id, ftype, N, fpt, ...
        numel(idxH), (any(strcmp(ftype,{'OuterRace','InnerRace'})) && fpt<=N) * numel(pick_even(fpt:N, cfg.xjtu.maxFaultFilesPerBearing)));
end
if ~isfolder(cfg.paths.cache), mkdir(cfg.paths.cache); end
save(hiFile, 'HI');

T = cell2table(rows, 'VariableNames', {'path','bearing','condIdx','speedHz','label','fileIndex','fpt','finalFault','split'});
T.condition = "C" + string(T.condIdx);
if cfg.debug
    T = T(T.split == "ref" | mod((1:height(T)).', 5) == 0, :);
end
end

function idx = pick_even(idx, nMax)
if numel(idx) > nMax
    idx = idx(unique(round(linspace(1, numel(idx), nMax))));
end
end
