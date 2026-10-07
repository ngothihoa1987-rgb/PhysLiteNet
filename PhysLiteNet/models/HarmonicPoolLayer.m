classdef HarmonicPoolLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    properties
        PoolMatrix
    end
    methods
        function layer = HarmonicPoolLayer(poolMatrix, name)
            layer.Name = name;
            layer.Description = "Harmonic pooling";
            layer.PoolMatrix = single(poolMatrix);
        end
        function Z = predict(layer, X)
            [~, L, C, N] = size(X);
            H = size(layer.PoolMatrix, 2);
            Xr = reshape(permute(X, [2 3 4 1]), L, C * N);
            Y = layer.PoolMatrix.' * Xr;
            Z = reshape(Y, 1, 1, H * C, N);
        end
    end
end
