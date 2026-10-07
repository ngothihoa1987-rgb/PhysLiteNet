function x = load_hust_signal(path, cfg)
txt = fileread(path);
k = regexp(txt, '\nData\s*\r?\n', 'end', 'once');
if isempty(k), error('Line "Data" not found in %s', path); end
v = sscanf(txt(k+1:end), '%f');
nCol = 5;
v = reshape(v(1:floor(numel(v)/nCol)*nCol), nCol, []).';
col = find(strcmpi(cfg.hust.channel, {'X','Y','Z'})) + 2;
x = v(:, col);
if cfg.fs ~= 25600
    x = resample(x, cfg.fs, 25600);
end
end
