function fr = refine_shaft_freq(x, fs, f0, relRange)
x = x(:) - mean(x);
nfft = 2^nextpow2(numel(x) * 4);
X = abs(fft(x .* hann(numel(x)), nfft));
df = fs / nfft;
cands = f0 * (1 + (relRange(1):1e-3:relRange(2)));
score = zeros(size(cands));
for h = 1:4
    idx = round(h * cands / df) + 1;
    s = max([X(idx) X(idx-1) X(idx+1)], [], 2);
    score = score + s.' / median(X(max(1,round(h*f0*0.8/df)):round(h*f0*1.2/df)));
end
[~, i] = max(score);
fr = cands(i);
end
