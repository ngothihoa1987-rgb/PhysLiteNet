function [x, fr] = load_signal(dataset, row, cfg)
switch upper(dataset)
    case 'PU'
        [x, fr] = load_pu_signal(row.path, cfg);
    case 'HUST'
        x = load_hust_signal(row.path, cfg);
        fr = row.speedHz;
    case 'XJTU'
        x = load_xjtu_signal(row.path, cfg);
        fr = row.speedHz;
end
end
