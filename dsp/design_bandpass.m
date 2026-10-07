function [sos, g] = design_bandpass(cfg)
[z, p, k] = butter(cfg.phys.filterOrder, cfg.phys.band / (cfg.fs/2), 'bandpass');
[sos, g] = zp2sos(z, p, k);
end
