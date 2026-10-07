clear; clc;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config();

DS = load_all_datasets(cfg);

eda_check_signatures(cfg, DS);

run_ml_baselines(cfg, DS);

run_dl_experiments(cfg, DS, cfg.models, ["physics" "physref"]);

run_dl_experiments(cfg, DS, cfg.ablations, "physics");

run_dl_experiments(cfg, DS, cfg.ablations, "physref");

run_dl_experiments(cfg, DS, cfg.models, "raw", 1:5);

analyze_results(cfg);
per_target_tests(cfg);
multi_record_eval(cfg, DS);
make_figures(cfg);
plot_tsne(cfg, DS, "PhysLiteNet", "physics");
plot_tsne(cfg, DS, "PhysLiteNet", "physref");
plot_tsne(cfg, DS, "ResNet1D18", "raw");
