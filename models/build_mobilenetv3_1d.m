function net = build_mobilenetv3_1d(inputSize, numClasses)
cfgB = [3  16  16 1 0 2
        3  72  24 0 0 2
        3  88  24 0 0 1
        5  96  40 1 1 2
        5 240  40 1 1 1
        5 240  40 1 1 1
        5 120  48 1 1 1
        5 144  48 1 1 1
        5 288  96 1 1 2
        5 576  96 1 1 1
        5 576  96 1 1 1];
net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'in')
    convolution2dLayer([1 3], 16, 'Stride', [1 2], 'Padding', 'same', 'Name', 'stem_conv')
    batchNormalizationLayer('Name', 'stem_bn')
    hswish_layer('stem_act')]);
last = 'stem_act'; Cin = 16;
for i = 1:size(cfgB, 1)
    [net, last] = add_bneck(net, last, sprintf('bn%d_', i), Cin, cfgB(i,:));
    Cin = cfgB(i, 3);
end
net = addLayers(net, [
    convolution2dLayer(1, 576, 'Name', 'last_conv')
    batchNormalizationLayer('Name', 'last_bn')
    hswish_layer('last_act')
    globalAveragePooling2dLayer('Name', 'gap')
    convolution2dLayer(1, 1024, 'Name', 'head_fc1')
    hswish_layer('head_act')
    dropoutLayer(0.2, 'Name', 'drop')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')]);
net = connectLayers(net, last, 'last_conv');
net = initialize(net);
end

function [net, outName] = add_bneck(net, inName, p, Cin, c)
k = c(1); E = c(2); Cout = c(3); useSE = c(4); hs = c(5); s = c(6);
act = @(nm) reluLayer('Name', nm);
if hs, act = @(nm) hswish_layer(nm); end
layers = [];
if E ~= Cin
    layers = [convolution2dLayer(1, E, 'Name', [p 'exp'])
              batchNormalizationLayer('Name', [p 'exp_bn'])
              act([p 'exp_act'])];
end
layers = [layers
          groupedConvolution2dLayer([1 k], 1, E, 'Stride', [1 s], 'Padding', 'same', 'Name', [p 'dw'])
          batchNormalizationLayer('Name', [p 'dw_bn'])
          act([p 'dw_act'])];
net = addLayers(net, layers);
net = connectLayers(net, inName, layers(1).Name);
last = [p 'dw_act'];
if useSE
    r = max(8, round(E / 4));
    net = addLayers(net, [
        globalAveragePooling2dLayer('Name', [p 'se_gap'])
        convolution2dLayer(1, r, 'Name', [p 'se_fc1'])
        reluLayer('Name', [p 'se_relu'])
        convolution2dLayer(1, E, 'Name', [p 'se_fc2'])
        functionLayer(@(X) min(max(X + 3, 0), 6) / 6, 'Name', [p 'se_hsig'], 'Acceleratable', true)]);
    net = addLayers(net, functionLayer(@(X, S) X .* S, 'NumInputs', 2, 'Name', [p 'se_mul'], 'Acceleratable', true));
    net = connectLayers(net, last, [p 'se_gap']);
    net = connectLayers(net, last, [p 'se_mul/in1']);
    net = connectLayers(net, [p 'se_hsig'], [p 'se_mul/in2']);
    last = [p 'se_mul'];
end
net = addLayers(net, [
    convolution2dLayer(1, Cout, 'Name', [p 'proj'])
    batchNormalizationLayer('Name', [p 'proj_bn'])]);
net = connectLayers(net, last, [p 'proj']);
outName = [p 'proj_bn'];
if s == 1 && Cin == Cout
    net = addLayers(net, additionLayer(2, 'Name', [p 'add']));
    net = connectLayers(net, [p 'proj_bn'], [p 'add/in1']);
    net = connectLayers(net, inName, [p 'add/in2']);
    outName = [p 'add'];
end
end

function L = hswish_layer(name)
L = functionLayer(@(X) X .* min(max(X + 3, 0), 6) / 6, 'Name', name, 'Acceleratable', true);
end
