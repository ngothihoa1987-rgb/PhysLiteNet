function DS = load_all_datasets(cfg)
DS.PU = add_reference_inputs(prepare_dataset("PU", cfg), cfg);
DS.targets = strings(0);
for t = ["HUST" "XJTU"]
    try
        DS.(t) = add_reference_inputs(prepare_dataset(t, cfg), cfg);
        DS.targets(end+1) = t;
    catch err
        warning('Skipping %s: %s', t, err.message);
    end
end
end
