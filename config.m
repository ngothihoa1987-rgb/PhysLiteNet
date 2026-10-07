function cfg = config()

cfg.root = fileparts(mfilename('fullpath'));
cfg.round = 2;

cfg.paths.PU   = fullfile(cfg.root, 'data_raw', 'PU');
cfg.paths.HUST = fullfile(cfg.root, 'data_raw', 'HUST');
cfg.paths.XJTU = fullfile(cfg.root, 'data_raw', 'XJTU-SY');
cfg.paths.cache   = fullfile(cfg.root, 'cache');
cfg.paths.results = fullfile(cfg.root, 'results');

cfg.fs      = 25600;
cfg.classes = ["Healthy" "OuterRace" "InnerRace"];
cfg.useParallel  = false;
cfg.forceRebuild = false;
cfg.debug        = false;

cfg.phys.band        = [1000 10000];
cfg.phys.filterOrder = 4;
cfg.phys.nRev        = 32;
cfg.phys.overlap     = 0.5;
cfg.phys.maxSegPerFile = 20;
cfg.phys.L           = 512;
cfg.phys.umin        = 0.1;
cfg.phys.umax        = 5.5;
cfg.phys.baseOrders  = ["shaft" "BPFO" "BPFI"];
cfg.phys.zeroPad     = 4;
cfg.phys.nHarm       = 5;

cfg.ref.xjtuFirstN = 5;
cfg.ref.floor      = 0.5;

cfg.raw.winLen = 2048;
cfg.raw.winPerFile = struct('PU', 16, 'HUST', 40, 'XJTU', 2);

cfg.speed.refine   = true;
cfg.speed.relRange = struct('HUST', [-0.05 0.01], 'XJTU', [-0.05 0.01]);

cfg.pu.healthy = ["K001" "K002" "K003" "K004" "K005" "K006"];
cfg.pu.outer   = ["KA01" "KA03" "KA05" "KA06" "KA07" "KA08" "KA09" ...
                  "KA04" "KA15" "KA16" "KA22" "KA30"];
cfg.pu.inner   = ["KI01" "KI03" "KI05" "KI07" "KI08" ...
                  "KI04" "KI14" "KI16" "KI17" "KI18" "KI21"];
cfg.pu.artificial = ["KA01" "KA03" "KA05" "KA06" "KA07" "KA08" "KA09" ...
                     "KI01" "KI03" "KI05" "KI07" "KI08"];
cfg.pu.valBearings  = ["K006" "KA05" "KA22" "KI05" "KI18"];
cfg.pu.testBearings = ["K004" "KA03" "KA15" "KI03" "KI14"];

cfg.hust.channel   = 'Z';
cfg.hust.includeVS = false;

cfg.xjtu.channel     = 1;
cfg.xjtu.speedHz     = [35 37.5 40];
cfg.xjtu.baselineFrac = 0.1;
cfg.xjtu.minBaseline = 5;
cfg.xjtu.kSigma      = 3;
cfg.xjtu.nConsec     = 3;
cfg.xjtu.healthyFrac = 0.8;
cfg.xjtu.maxHealthyFilesPerBearing = 60;
cfg.xjtu.maxFaultFilesPerBearing   = 120;

cfg.ml.svmC    = 1;
cfg.ml.rfTrees = 200;
cfg.ml.inputs  = ["features" "featuresref"];

cfg.aug.enable        = true;
cfg.aug.pJitter       = 0.5;  cfg.aug.jitter = 0.02;
cfg.aug.pShaftHarm    = 0.5;  cfg.aug.maxShaftHarm = 6;
cfg.aug.ampShaft      = [0.5 6];
cfg.aug.pSpur         = 0.5;  cfg.aug.nSpur = [1 3];
cfg.aug.ampSpur       = [1 8];
cfg.aug.peakWidthBins = 1.5;
cfg.aug.noise         = 0.1;

cfg.models = ["PhysLiteNet" "ResNet1D18" "MobileNetV3_1D" "ShuffleNetV2_1D"];
cfg.inputs = ["physics" "physref" "raw"];
cfg.ablations = ["PhysLiteNet_noPrior" "PhysLiteNet_noAug"];
cfg.seeds  = 1:10;
cfg.train.maxEpochs   = 60;
cfg.train.miniBatch   = 128;
cfg.train.learnRate   = 1e-3;
cfg.train.l2          = 1e-4;
cfg.train.patience    = 10;
cfg.train.execEnv     = 'auto';
cfg.train.verbose     = true;

cfg.multi.windows = [1 5 10];

if cfg.debug
    cfg.paths.cache   = fullfile(cfg.root, 'cache_debug');
    cfg.paths.results = fullfile(cfg.root, 'results_debug');
    cfg.train.maxEpochs = 3;
    cfg.seeds = 1;
end
end
