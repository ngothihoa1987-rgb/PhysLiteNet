function o = bearing_orders(dataset)
switch upper(dataset)
    case 'PU'
        n = 8; d = 6.75; D = 28.55; alpha = 0;
    case 'HUST'
        n = 9; d = 7.94; D = 38.52; alpha = 0;
    case 'XJTU'
        n = 8; d = 7.92; D = 34.55; alpha = 0;
    otherwise
        error('Unknown dataset %s', dataset);
end
r = d / D * cosd(alpha);
o.shaft = 1;
o.BPFO  = n / 2 * (1 - r);
o.BPFI  = n / 2 * (1 + r);
o.BSF   = D / (2 * d) * (1 - r^2);
o.FTF   = 0.5 * (1 - r);
end
