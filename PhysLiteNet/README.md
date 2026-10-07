# PhysLiteNet

MATLAB code for the paper

> Ngo Thi Hoa. *PhysLiteNet: A physics-guided lightweight network for cross-machine bearing fault diagnosis without target-domain fault data.* 

The model is trained on the Paderborn University (PU) bearing dataset and tested on the HUST and XJTU-SY datasets without using target labels.

## Requirements

- MATLAB R2024a or later
- Deep Learning Toolbox
- Signal Processing Toolbox

The SVM, random forest, Friedman and Wilcoxon tests and t-SNE are implemented in `ml/`, so the Statistics and Machine Learning Toolbox is not required.

## Data

The datasets are not included. Download them from their providers:

- **PU**: KAt-DataCenter, Paderborn University (Lessmeier et al., 2016)
- **HUST**: https://github.com/CHAOZHAO-1/HUSTbearing-dataset (Zhao et al., 2024)
- **XJTU-SY**: https://biaowang.tech/xjtu-sy-bearing-datasets (Wang et al., 2020)

Place them as follows, or edit the three paths at the top of `config.m`:

```
data_raw/PU/        K001 ... KI21 folders (.mat files)
data_raw/HUST/      .xls files
data_raw/XJTU-SY/   35Hz12kN, 37.5Hz11kN, 40Hz10kN folders
```

## Usage

```matlab
main_run_all
```

The script builds the feature cache, trains all models (10 seeds; raw-signal models 5 seeds) and writes the results, statistical tests and figures to `results/`. Runs that already exist are skipped, so the script can be stopped and restarted.

| Folder | Content |
|---|---|
| `data/` | file lists, signal loading, XJTU-SY labelling, cache |
| `dsp/` | band-pass filter, shaft-speed refinement, fault-order-aligned envelope spectrum, features |
| `models/` | PhysLiteNet (harmonic prior attention, harmonic pooling) and baselines |
| `train/` | training loop, augmentation, healthy-reference normalisation, experiments |
| `ml/` | SVM, random forest, statistical tests, t-SNE |
| `analysis/` | result tables, statistical tests, multi-record evaluation, figures |

## References

- Lessmeier C, Kimotho JK, Zimmer D and Sextro W (2016) Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification. *PHM Society European Conference*.
- Zhao C, Zio E and Shen W (2024) Domain generalization for cross-domain fault diagnosis: An application-oriented perspective and a benchmark study. *Reliability Engineering & System Safety* 245: 109964.
- Wang B, Lei Y, Li N and Li N (2020) A hybrid prognostics approach for estimating remaining useful life of rolling element bearings. *IEEE Transactions on Reliability* 69(1): 401–412.

## License

MIT
