function T = hust_file_list(cfg)
files = dir(fullfile(cfg.paths.HUST, '*.xls'));
if isempty(files), error('No .xls files found in %s', cfg.paths.HUST); end

rows = {};
for i = 1:numel(files)
    nm = files(i).name;
    tok = regexp(nm, '^(0\.5X_)?([HIOBC])_(.+?)\.xls$', 'tokens', 'once', 'ignorecase');
    if isempty(tok), continue; end
    severity = "severe"; if ~isempty(tok{1}), severity = "medium"; end
    state = upper(string(tok{2}));
    spd = string(tok{3});
    isVS = startsWith(upper(spd), "VS");
    if isVS
        if ~cfg.hust.includeVS, continue; end
        speedHz = NaN;
    else
        speedHz = str2double(regexprep(spd, '[^\d.]', ''));
    end
    switch state
        case "H", label = "Healthy"; severity = "none";
        case "O", label = "OuterRace";
        case "I", label = "InnerRace";
        otherwise, continue;
    end
    rows(end+1,:) = {string(fullfile(files(i).folder, nm)), state, severity, speedHz, label, isVS};
end
T = cell2table(rows, 'VariableNames', {'path','state','severity','speedHz','label','isVS'});
T.bearing = T.state + "_" + T.severity;
T.condition = string(T.speedHz) + "Hz";

if cfg.debug
    T = T(ismember(T.speedHz, [20 40 80]), :);
end
end
