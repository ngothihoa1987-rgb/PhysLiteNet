function [x, fr] = load_pu_signal(path, cfg)
s = load(path);
fn = fieldnames(s);
d = s.(fn{1});
names = {d.Y.Name};
v = d.Y(strcmp(names, 'vibration_1')).Data;
x = resample(double(v(:)), 2, 5);
sp = d.Y(strcmp(names, 'speed')).Data;
fr = mean(double(sp)) / 60;
if cfg.fs ~= 25600
    x = resample(x, cfg.fs, 25600);
end
end
