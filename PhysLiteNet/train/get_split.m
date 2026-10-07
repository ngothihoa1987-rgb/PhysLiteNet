function [X, Y, meta] = get_split(D, inputType, split)
switch inputType
    case "physics",     X = D.Xphys;    Y = D.Yphys; meta = D.metaPhys;
    case "physref",     X = D.XphysRef; Y = D.Yphys; meta = D.metaPhys;
    case "features",    X = D.feat;     Y = D.Yphys; meta = D.metaPhys;
    case "featuresref", X = D.featRef;  Y = D.Yphys; meta = D.metaPhys;
    case "raw",         X = D.Xraw;     Y = D.Yraw;  meta = D.metaRaw;
    otherwise, error('Unknown input type %s', inputType);
end
if nargin < 3, split = "all"; end
if split == "all"
    k = meta.split ~= "ref";
else
    k = meta.split == split;
end
if any(inputType == ["features" "featuresref"])
    X = X(k, :);
else
    X = X(:, :, :, k);
end
Y = Y(k); meta = meta(k, :);
end
