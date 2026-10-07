function x = load_xjtu_signal(path, cfg)
v = readmatrix(path, 'NumHeaderLines', 1);
x = v(:, cfg.xjtu.channel);
x = x(~isnan(x));
if cfg.fs ~= 25600
    x = resample(x, cfg.fs, 25600);
end
end
