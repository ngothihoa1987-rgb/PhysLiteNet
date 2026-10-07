function eda_check_signatures(cfg, DS)
figDir = fullfile(cfg.paths.results, 'figures'); if ~isfolder(figDir), mkdir(figDir); end
names = ["PU" DS.targets];
dispName = containers.Map({'PU','HUST','XJTU'}, {'PU','HUST','XJTU-SY'});
chName = containers.Map({'shaft','BPFO','BPFI'}, {'shaft channel','BPFO channel','BPFI channel'});
clsName = {'Healthy', 'Outer race', 'Inner race'};
e = linspace(cfg.phys.umin, cfg.phys.umax, cfg.phys.L + 1); u = (e(1:end-1) + e(2:end)) / 2;
col = lines(numel(cfg.classes));
nC = numel(cfg.phys.baseOrders);
f = figure('Color', 'w', 'Units', 'inches', 'Position', [0.5 0.5 11 2.9 * numel(names)]);
for i = 1:numel(names)
    D = DS.(names(i));
    for c = 1:nC
        subplot(numel(names), nC, (i-1) * nC + c); hold on
        for k = 1:numel(cfg.classes)
            m = D.Yphys == cfg.classes(k);
            if ~any(m), continue; end
            plot(u, squeeze(mean(D.Xphys(1, :, c, m), 4)), 'Color', col(k,:), 'LineWidth', 0.9);
        end
        xline(1:cfg.phys.nHarm, ':k');
        xlim([cfg.phys.umin cfg.phys.umax]); grid on; box on
        set(gca, 'FontSize', 9);
        title(sprintf('%s, %s', dispName(char(names(i))), chName(char(cfg.phys.baseOrders(c)))));
        if c == 1, ylabel('log(1 + E / median)'); end
        if i == numel(names), xlabel('Normalised order u'); end
    end
end
legend(clsName, 'Location', 'best');
exportgraphics(f, fullfile(figDir, 'eda_aligned_envelope_spectra.png'), 'Resolution', 300);
set(f, 'PaperUnits', 'inches'); p = get(f, 'Position');
set(f, 'PaperSize', p(3:4), 'PaperPosition', [0 0 p(3:4)]);
print(f, fullfile(figDir, 'eda_aligned_envelope_spectra.pdf'), '-dpdf', '-vector');
close(f);

for n = names
    fprintf('%-5s physics: %s | raw: %s\n', n, ...
        strjoin(string(countcats(DS.(n).Yphys)).', '/'), strjoin(string(countcats(DS.(n).Yraw)).', '/'));
end
end
