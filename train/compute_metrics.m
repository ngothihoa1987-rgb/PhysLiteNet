function m = compute_metrics(yTrue, scores, classes, fileId)
classes = cellstr(classes);
[~, ip] = max(scores, [], 2);
yPred = categorical(classes(ip), classes);
yPred = yPred(:);
m = basic_metrics(yTrue, yPred, classes);
m.yPred = yPred;
if nargin >= 4 && ~isempty(fileId)
    [g, ~] = findgroups(fileId);
    sf = splitapply(@(s) mean(s, 1), scores, g);
    yf = splitapply(@(y) y(1), yTrue, g);
    [~, ipf] = max(sf, [], 2);
    mf = basic_metrics(yf, categorical(classes(ipf), classes).', classes);
    m.accFile = mf.acc; m.f1File = mf.f1; m.baccFile = mf.bacc; m.cmFile = mf.cm;
end
end

function m = basic_metrics(yTrue, yPred, classes)
yTrue = yTrue(:); yPred = yPred(:);
K = numel(classes);
it = double(categorical(cellstr(yTrue), classes));
ip = double(categorical(cellstr(yPred), classes));
ok = ~isnan(it) & ~isnan(ip);
cm = accumarray([it(ok) ip(ok)], 1, [K K]);
present = sum(cm, 2) > 0;
tp = diag(cm);
recall = tp ./ max(sum(cm, 2), 1);
prec   = tp ./ max(sum(cm, 1).', 1);
f1 = 2 * prec .* recall ./ max(prec + recall, eps);
m.acc  = sum(tp) / max(sum(cm(:)), 1);
m.bacc = mean(recall(present));
m.f1   = mean(f1(present));
m.cm   = cm;
end
