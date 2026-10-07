function per_target_tests(cfg)
T = readtable(fullfile(cfg.paths.results, 'all_runs.csv'), 'TextType', 'string');
sets = {"PhysLiteNet (physref)", ["RF (featuresref)" "SVM (featuresref)" "ResNet1D18 (physref)" "MobileNetV3_1D (physref)" "ShuffleNetV2_1D (physref)" "PhysLiteNet_noPrior (physref)" "PhysLiteNet_noAug (physref)"];
        "PhysLiteNet (physics)", ["RF (features)" "SVM (features)" "ResNet1D18 (physics)" "MobileNetV3_1D (physics)" "ShuffleNetV2_1D (physics)" "PhysLiteNet_noPrior (physics)" "PhysLiteNet_noAug (physics)"]};
for s = 1:2
  ref = sets{s,1}; fprintf('\n===== %s =====\n', ref);
  for tg = ["HUST" "XJTU"]
    others = sets{s,2}; k = numel(others); p = zeros(1,k); d = p; n = p; w = p;
    for i = 1:k
      a = T(T.method == ref & T.target == tg, :); b = T(T.method == others(i) & T.target == tg, :);
      [sd, ia, ib] = intersect(a.seed, b.seed);
      d(i) = 100*mean(a.f1(ia) - b.f1(ib)); n(i) = numel(sd);
      p(i) = stat_signrank(a.f1(ia), b.f1(ib));
      w(i) = sum(a.f1(ia) > b.f1(ib));
    end
    [ps, o] = sort(p); ph = zeros(1,k); run = 0;
    for j = 1:k, run = max(run, min(1, (k-j+1)*ps(j))); ph(o(j)) = run; end
    fprintf('-- %s\n', tg);
    for i = 1:k, fprintf('  vs %-32s dF1 %+6.1f  wins %2d/%d  p %.4f  pHolm %.4f\n', others(i), d(i), w(i), n(i), p(i), ph(i)); end
  end
end
end
