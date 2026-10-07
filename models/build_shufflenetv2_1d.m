function net = build_shufflenetv2_1d(inputSize, numClasses)
stageC = [48 96 192]; rep = [4 8 4];
net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'in')
    convolution2dLayer([1 3], 24, 'Stride', [1 2], 'Padding', 'same', 'Name', 'stem_conv')
    batchNormalizationLayer('Name', 'stem_bn')
    reluLayer('Name', 'stem_relu')
    maxPooling2dLayer([1 3], 'Stride', [1 2], 'Padding', 'same', 'Name', 'stem_pool')]);
last = 'stem_pool'; Cin = 24;
for s = 1:3
    for r = 1:rep(s)
        p = sprintf('s%du%d_', s, r);
        if r == 1
            [net, last] = add_down_unit(net, last, p, Cin, stageC(s));
        else
            [net, last] = add_basic_unit(net, last, p, stageC(s));
        end
        Cin = stageC(s);
    end
end
net = addLayers(net, [
    convolution2dLayer(1, 1024, 'Name', 'conv5')
    batchNormalizationLayer('Name', 'conv5_bn')
    reluLayer('Name', 'conv5_relu')
    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')]);
net = connectLayers(net, last, 'conv5');
net = initialize(net);
end

function [net, outName] = add_basic_unit(net, inName, p, C)
h = C / 2;
net = addLayers(net, functionLayer(@(X) X(:, :, 1:h, :), 'Name', [p 'split1'], 'Acceleratable', true));
net = addLayers(net, [
    functionLayer(@(X) X(:, :, h+1:end, :), 'Name', [p 'split2'], 'Acceleratable', true)
    convolution2dLayer(1, h, 'Name', [p 'pw1'])
    batchNormalizationLayer('Name', [p 'bn1'])
    reluLayer('Name', [p 'relu1'])
    groupedConvolution2dLayer([1 3], 1, h, 'Padding', 'same', 'Name', [p 'dw'])
    batchNormalizationLayer('Name', [p 'bn2'])
    convolution2dLayer(1, h, 'Name', [p 'pw2'])
    batchNormalizationLayer('Name', [p 'bn3'])
    reluLayer('Name', [p 'relu2'])]);
net = addLayers(net, [
    depthConcatenationLayer(2, 'Name', [p 'cat'])
    functionLayer(@(X) channel_shuffle(X, 2), 'Name', [p 'shuffle'], 'Acceleratable', true)]);
net = connectLayers(net, inName, [p 'split1']);
net = connectLayers(net, inName, [p 'split2']);
net = connectLayers(net, [p 'split1'], [p 'cat/in1']);
net = connectLayers(net, [p 'relu2'], [p 'cat/in2']);
outName = [p 'shuffle'];
end

function [net, outName] = add_down_unit(net, inName, p, Cin, Cout)
h = Cout / 2;
net = addLayers(net, [
    groupedConvolution2dLayer([1 3], 1, Cin, 'Stride', [1 2], 'Padding', 'same', 'Name', [p 'b1_dw'])
    batchNormalizationLayer('Name', [p 'b1_bn1'])
    convolution2dLayer(1, h, 'Name', [p 'b1_pw'])
    batchNormalizationLayer('Name', [p 'b1_bn2'])
    reluLayer('Name', [p 'b1_relu'])]);
net = addLayers(net, [
    convolution2dLayer(1, h, 'Name', [p 'b2_pw1'])
    batchNormalizationLayer('Name', [p 'b2_bn1'])
    reluLayer('Name', [p 'b2_relu1'])
    groupedConvolution2dLayer([1 3], 1, h, 'Stride', [1 2], 'Padding', 'same', 'Name', [p 'b2_dw'])
    batchNormalizationLayer('Name', [p 'b2_bn2'])
    convolution2dLayer(1, h, 'Name', [p 'b2_pw2'])
    batchNormalizationLayer('Name', [p 'b2_bn3'])
    reluLayer('Name', [p 'b2_relu2'])]);
net = addLayers(net, [
    depthConcatenationLayer(2, 'Name', [p 'cat'])
    functionLayer(@(X) channel_shuffle(X, 2), 'Name', [p 'shuffle'], 'Acceleratable', true)]);
net = connectLayers(net, inName, [p 'b1_dw']);
net = connectLayers(net, inName, [p 'b2_pw1']);
net = connectLayers(net, [p 'b1_relu'], [p 'cat/in1']);
net = connectLayers(net, [p 'b2_relu2'], [p 'cat/in2']);
outName = [p 'shuffle'];
end
