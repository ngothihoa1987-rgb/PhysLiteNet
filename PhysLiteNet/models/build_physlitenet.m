function net = build_physlitenet(inputSize, numClasses, cfg, usePrior)
if nargin < 4, usePrior = true; end
L = inputSize(2); Cin = inputSize(3);
widths = [16 24 32 48];
stemStride = 1; if L > 1024, stemStride = 4; end

net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'in')
    convolution2dLayer([1 5], widths(1), 'Stride', [1 stemStride], 'Padding', 'same', 'Name', 'stem_conv')
    batchNormalizationLayer('Name', 'stem_bn')
    reluLayer('Name', 'stem_relu')]);
last = 'stem_relu';
Ls = ceil(L / stemStride);
for s = 1:3
    [net, last] = add_msds_block(net, last, sprintf('b%d_', s), widths(s), widths(s+1), 2);
    Ls = ceil(Ls / 2);
end
C = widths(end);

if usePrior
    [T, W] = harmonic_template(Ls, cfg);
    net = addLayers(net, HarmonicPriorAttentionLayer(C, T, 'hpa'));
    net = connectLayers(net, last, 'hpa');
    last = 'hpa';
end

net = addLayers(net, globalAveragePooling2dLayer('Name', 'gap'));
net = connectLayers(net, last, 'gap');
head = [dropoutLayer(0.2, 'Name', 'drop')
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')];
if usePrior
    net = addLayers(net, HarmonicPoolLayer(W, 'hpool'));
    net = connectLayers(net, last, 'hpool');
    net = addLayers(net, [depthConcatenationLayer(2, 'Name', 'cat'); head]);
    net = connectLayers(net, 'gap', 'cat/in1');
    net = connectLayers(net, 'hpool', 'cat/in2');
else
    net = addLayers(net, head);
    net = connectLayers(net, 'gap', 'drop');
end
net = initialize(net);
end

function [net, outName] = add_msds_block(net, inName, p, Cin, Cout, stride)
ks = [3 7 15];
net = addLayers(net, additionLayer(numel(ks), 'Name', [p 'msum']));
for i = 1:numel(ks)
    nm = sprintf('%sdw%d', p, ks(i));
    net = addLayers(net, groupedConvolution2dLayer([1 ks(i)], 1, Cin, ...
        'Stride', [1 stride], 'Padding', 'same', 'Name', nm));
    net = connectLayers(net, inName, nm);
    net = connectLayers(net, nm, sprintf('%smsum/in%d', p, i));
end
net = addLayers(net, [
    batchNormalizationLayer('Name', [p 'bn1'])
    reluLayer('Name', [p 'relu1'])
    convolution2dLayer(1, Cout, 'Name', [p 'pw'])
    batchNormalizationLayer('Name', [p 'bn2'])
    additionLayer(2, 'Name', [p 'add'])
    reluLayer('Name', [p 'out'])]);
net = connectLayers(net, [p 'msum'], [p 'bn1']);
net = addLayers(net, [
    convolution2dLayer(1, Cout, 'Stride', [1 stride], 'Name', [p 'sc_conv'])
    batchNormalizationLayer('Name', [p 'sc_bn'])]);
net = connectLayers(net, inName, [p 'sc_conv']);
net = connectLayers(net, [p 'sc_bn'], [p 'add/in2']);
outName = [p 'out'];
end
