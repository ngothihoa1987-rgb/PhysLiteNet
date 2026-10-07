function net = build_model(modelName, inputSize, numClasses, cfg, inputType)
hasAxis = any(string(inputType) == ["physics" "physref"]);
switch string(modelName)
    case {"PhysLiteNet", "PhysLiteNet_noAug"}
        net = build_physlitenet(inputSize, numClasses, cfg, hasAxis);
    case "PhysLiteNet_noPrior"
        net = build_physlitenet(inputSize, numClasses, cfg, false);
    case "ResNet1D18"
        net = build_resnet1d18(inputSize, numClasses);
    case "MobileNetV3_1D"
        net = build_mobilenetv3_1d(inputSize, numClasses);
    case "ShuffleNetV2_1D"
        net = build_shufflenetv2_1d(inputSize, numClasses);
    otherwise
        error('Unknown model %s', modelName);
end
end
