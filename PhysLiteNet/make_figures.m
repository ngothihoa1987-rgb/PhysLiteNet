function make_figures(cfg)
R = [cfg.paths.results filesep];
O = [fullfile(cfg.paths.results, 'figures') filesep];
if ~isfolder(O), mkdir(O); end
S = readtable([R 'summary_mean_std.csv'], 'TextType', 'string');
C = readtable([R 'complexity.csv'], 'TextType', 'string');
Mr = readtable([R 'multi_record.csv'], 'TextType', 'string');
get = @(m,t,f) S.(f)(S.method == m & S.target == t);
tg = ["PU_test" "HUST" "XJTU"];
col = [0.20 0.45 0.70; 0.85 0.45 0.15; 0.35 0.65 0.30];

setA = {["SVM (features)" "RF (features)" "ResNet1D18 (physics)" "MobileNetV3_1D (physics)" "ShuffleNetV2_1D (physics)" "PhysLiteNet (physics)"], ...
        ["SVM (featuresref)" "RF (featuresref)" "ResNet1D18 (physref)" "MobileNetV3_1D (physref)" "ShuffleNetV2_1D (physref)" "PhysLiteNet (physref)"]};
lbl = ["SVM" "RF" "ResNet1D-18" "MobileNetV3-1D" "ShuffleNetV2-1D" "PhysLiteNet"];
ttl = ["(a) Strict zero-shot" "(b) With healthy reference"];
f = figure('Color','w','Units','inches','Position',[0.5 0.5 11 4]);
for p = 1:2
  subplot(1,2,p); hold on
  M = zeros(6,3); E = M;
  for i = 1:6, for t = 1:3, M(i,t) = 100*get(setA{p}(i), tg(t), 'mean_f1'); E(i,t) = 100*get(setA{p}(i), tg(t), 'std_f1'); end, end
  b = bar(M, 'grouped', 'EdgeColor', 'none');
  for t = 1:3, b(t).FaceColor = col(t,:); errorbar(b(t).XEndPoints, M(:,t), E(:,t), 'k.', 'LineWidth', 0.8, 'CapSize', 3); end
  set(gca, 'XTick', 1:6, 'XTickLabel', lbl, 'XTickLabelRotation', 25, 'FontSize', 9);
  ylim([0 95]); ylabel('Macro-F1 (%)'); title(ttl(p)); grid on; box on
  if p == 1, legend(b, ["PU-test (in-domain)" "HUST" "XJTU-SY"], 'Location', 'northwest'); end
end
savepdf(f, [O 'fig_main_bars.pdf']);

items = { "PhysLiteNet (physref)",'s',1.25,0,'left'; "PhysLiteNet (physics)",'o',1.25,0,'left'; "PhysLiteNet (raw)",'^',1.25,0,'left';
          "ShuffleNetV2_1D (physref)",'s',0.80,1.2,'right'; "ShuffleNetV2_1D (physics)",'o',0.80,0,'right'; "ShuffleNetV2_1D (raw)",'^',1.25,-3.0,'left';
          "MobileNetV3_1D (physref)",'s',0.80,3.0,'right'; "MobileNetV3_1D (physics)",'o',1.25,-3.0,'left'; "MobileNetV3_1D (raw)",'^',0.80,2.5,'right';
          "ResNet1D18 (physref)",'s',1.25,3.0,'left'; "ResNet1D18 (physics)",'o',1.25,0.8,'left'; "ResNet1D18 (raw)",'^',1.25,0,'left'};
clr = containers.Map({'PhysLiteNet','ResNet1D18','MobileNetV3_1D','ShuffleNetV2_1D'}, {[0.80 0.20 0.20],[0.20 0.45 0.70],[0.35 0.65 0.30],[0.55 0.35 0.70]});
f = figure('Color','w','Units','inches','Position',[0.5 0.5 8 4.6]); hold on
for i = 1:size(items,1)
  m = items{i,1}; base = char(extractBefore(m, " ("));
  y = 100*mean([get(m,"HUST",'mean_f1') get(m,"XJTU",'mean_f1')]); x = C.params(C.method == m);
  if items{i,3} ~= 1 || items{i,4} ~= 0, plot([x x*items{i,3}], [y y+items{i,4}], '-', 'Color', [0.6 0.6 0.6]); end
  scatter(x, y, 60, clr(base), items{i,2}, 'filled', 'MarkerEdgeColor', 'k');
  lab = nicename(m);
  text(x*items{i,3}, y + items{i,4}, sprintf(' %s %.1f ', lab, y), 'FontSize', 7.5, 'HorizontalAlignment', items{i,5});
end
set(gca, 'XScale', 'log', 'FontSize', 9); xlim([3e3 4e7]); ylim([15 78]);
xlabel('Number of parameters (log scale)'); ylabel('Mean target macro-F1 (%), HUST and XJTU-SY');
h1 = scatter(nan,nan,50,'k','s','filled'); h2 = scatter(nan,nan,50,'k','o','filled'); h3 = scatter(nan,nan,50,'k','^','filled');
legend([h1 h2 h3], ["Aligned + healthy reference" "Aligned (zero-shot)" "Raw signal"], 'Location', 'northeast'); grid on; box on
savepdf(f, [O 'fig_tradeoff.pdf']);

mm = ["PhysLiteNet (physref)" "RF (featuresref)" "ResNet1D18 (physref)" "MobileNetV3_1D (physref)" "PhysLiteNet (physics)" "RF (features)"];
ls = {'-','-','-','-','--','--'}; mk = 'osd^os';
cc = [0.80 0.20 0.20; 0.40 0.40 0.40; 0.20 0.45 0.70; 0.35 0.65 0.30; 0.80 0.20 0.20; 0.40 0.40 0.40];
f = figure('Color','w','Units','inches','Position',[0.5 0.5 11 4.2]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for t = 1:3
  nexttile; hold on
  for i = 1:numel(mm)
    k = Mr.method == mm(i) & Mr.target == tg(t); [w, o] = sort(Mr.W(k)); y = 100*Mr.mean_f1(k); e = 100*Mr.std_f1(k);
    errorbar(w, y(o), e(o), ls{i}, 'Marker', mk(i), 'Color', cc(i,:), 'LineWidth', 1.2, 'MarkerFaceColor', cc(i,:), 'CapSize', 3);
  end
  set(gca, 'XTick', [1 5 10], 'FontSize', 9); xlim([0 11]); grid on; box on
  xlabel('Records per decision, W'); if t == 1, ylabel('Macro-F1 (%)'); end
  title(strrep(strrep(tg(t), '_', '-'), 'XJTU', 'XJTU-SY'));
end
lg = legend(["PhysLiteNet (healthy ref.)" "RF (healthy ref.)" "ResNet1D-18 (healthy ref.)" "MobileNetV3-1D (healthy ref.)" "PhysLiteNet (zero-shot)" "RF (zero-shot)"], 'NumColumns', 3, 'FontSize', 8); lg.Layout.Tile = 'south';
savepdf(f, [O 'fig_multirecord.pdf']);

cls = {'Healthy','Outer race','Inner race'};
cases = {'PhysLiteNet__physics','XJTU','(a) XJTU-SY, strict zero-shot','cm_physics_XJTU';
         'PhysLiteNet__physref','XJTU','(b) XJTU-SY, healthy reference','cm_physref_XJTU';
         'PhysLiteNet__physref','HUST','(c) HUST, healthy reference','cm_physref_HUST'};
for i = 1:size(cases,1)
  cm = 0;
  for s = cfg.seeds, L = load(sprintf('%sruns/%s__s%d.mat', R, cases{i,1}, s), 'r'); cm = cm + L.r.(cases{i,2}).cm; end
  f = figure('Color','w','Units','inches','Position',[0.5 0.5 3.6 3.2]);
  h = confusionchart(cm, cls, 'RowSummary', 'row-normalized', 'Title', cases{i,3}, 'FontSize', 8);
  h.Normalization = 'row-normalized';
  savepdf(f, [O cases{i,4} '.pdf']);
  fprintf('%s recall: %s\n', cases{i,4}, mat2str(round(100*diag(cm)'./sum(cm,2)'),3));
end
end

function s = nicename(m)
m = string(m); s = strrep(strrep(extractBefore(m, " ("), '_1D', '-1D'), '1D18', '1D-18');
if endsWith(m, "ref)"), s = s + " (healthy ref.)";
elseif endsWith(m, "(raw)"), s = s + " (raw)";
elseif endsWith(m, "(features)"), s = s + " (zero-shot)";
else, s = s + " (aligned)"; end
end

function savepdf(f, file)
set(f, 'PaperUnits', 'inches'); p = get(f, 'Position');
set(f, 'PaperSize', p(3:4), 'PaperPosition', [0 0 p(3:4)]);
print(f, file, '-dpdf', '-vector'); close(f);
end
