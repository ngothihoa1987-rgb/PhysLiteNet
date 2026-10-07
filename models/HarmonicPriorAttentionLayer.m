classdef HarmonicPriorAttentionLayer < nnet.layer.Layer
    properties
        Template
    end
    properties (Learnable)
        Weights
        Bias
        Gamma
    end
    methods
        function layer = HarmonicPriorAttentionLayer(numChannels, template, name)
            layer.Name = name;
            layer.Description = "Harmonic prior attention";
            layer.Template = reshape(single(template), 1, []);
            layer.Weights = single(0.1 * randn(1, 1, numChannels) / sqrt(numChannels));
            layer.Bias  = single(0);
            layer.Gamma = single(1);
        end
        function Z = predict(layer, X)
            s = sum(X .* layer.Weights, 3) + layer.Bias + layer.Gamma .* layer.Template;
            Z = X .* (1 + sigmoid(s));
        end
    end
end
