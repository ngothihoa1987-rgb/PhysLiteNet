function net = build_resnet1d18(inputSize, numClasses)
net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'in')
    convolution2dLayer([1 7], 64, 'Stride', [1 2], 'Padding', 'same', 'Name', 'stem_conv')
    batchNormalizationLayer('Name', 'stem_bn')
    reluLayer('Name', 'stem_relu')
    maxPooling2dLayer([1 3], 'Stride', [1 2], 'Padding', 'same', 'Name', 'stem_pool')]);
last = 'stem_pool'; Cin = 64;
widths = [64 128 256 512];
for s = 1:4
    for b = 1:2
        stride = 1; if s > 1 && b == 1, stride = 2; end
        [net, last] = add_basic_block(net, last, sprintf('s%db%d_', s, b), Cin, widths(s), stride);
        Cin = widths(s);
    end
end
net = addLayers(net, [
    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')]);
net = connectLayers(net, last, 'gap');
net = initialize(net);
end

function [net, outName] = add_basic_block(net, inName, p, Cin, Cout, stride)
net = addLayers(net, [
    convolution2dLayer([1 3], Cout, 'Stride', [1 stride], 'Padding', 'same', 'Name', [p 'conv1'])
    batchNormalizationLayer('Name', [p 'bn1'])
    reluLayer('Name', [p 'relu1'])
    convolution2dLayer([1 3], Cout, 'Padding', 'same', 'Name', [p 'conv2'])
    batchNormalizationLayer('Name', [p 'bn2'])
    additionLayer(2, 'Name', [p 'add'])
    reluLayer('Name', [p 'out'])]);
net = connectLayers(net, inName, [p 'conv1']);
if stride > 1 || Cin ~= Cout
    net = addLayers(net, [
        convolution2dLayer(1, Cout, 'Stride', [1 stride], 'Name', [p 'ds_conv'])
        batchNormalizationLayer('Name', [p 'ds_bn'])]);
    net = connectLayers(net, inName, [p 'ds_conv']);
    net = connectLayers(net, [p 'ds_bn'], [p 'add/in2']);
else
    net = connectLayers(net, inName, [p 'add/in2']);
end
outName = [p 'out'];
end
