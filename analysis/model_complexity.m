function c = model_complexity(net, inputSize, nRuns)
if nargin < 3, nRuns = 100; end
c.params = sum(cellfun(@numel, net.Learnables.Value));

X = dlarray(zeros([inputSize 1], 'single'), 'SSCB');
isConv = arrayfun(@(l) isa(l, 'nnet.cnn.layer.Convolution2DLayer') || ...
    isa(l, 'nnet.cnn.layer.GroupedConvolution2DLayer') || ...
    isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), net.Layers);
Ls = net.Layers(isConv);
outs = cell(1, numel(Ls));
[outs{:}] = predict(net, X, 'Outputs', {Ls.Name});
macs = 0;
for i = 1:numel(Ls)
    l = Ls(i); o = outs{i};
    nOut = numel(o);
    if isa(l, 'nnet.cnn.layer.Convolution2DLayer')
        macs = macs + nOut * prod(l.FilterSize) * l.NumChannels;
    elseif isa(l, 'nnet.cnn.layer.GroupedConvolution2DLayer')
        macs = macs + nOut * prod(l.FilterSize) * l.NumChannelsPerGroup;
    else
        macs = macs + l.InputSize * l.OutputSize;
    end
end
c.MMACs = macs / 1e6;

for k = 1:5, predict(net, X); end
t = zeros(nRuns, 1);
for k = 1:nRuns
    t0 = tic; predict(net, X); t(k) = toc(t0);
end
c.latencyMs = median(t) * 1e3;
s = whos('net'); c.sizeMB = s.bytes / 2^20;
end
