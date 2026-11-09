# ML-driven PVT-variation-aware leakage estimation

Digital VLSI Design course project: estimate the leakage current of CMOS circuits under process, voltage and temperature
variation with machine-learning transistor models instead of full SPICE simulation. The assignment is in
[`DVD_ML_PROJECT.pdf`](DVD_ML_PROJECT.pdf).

## Assigned configuration

| Item | Value |
|---|---|
| Model file | `45nm_MGK.pm` (PTM 45 nm metal gate / high-k, BSIM4 level 54) |
| Nominal VDD | 0.9 V (0.81 – 0.99 V in the samples) |
| Channel length / widths | L = 45 nm, W = 45 – 360 nm (1 – 8 × Lmin) |
| Test circuit | 3-bit carry-lookahead (CLA) adder |
| Random seed | 12175305 (every script and notebook) |
| Samples per transistor case | 20,000 (80 / 10 / 10 split → 16,000 / 2,000 / 2,000) |
| Simulator | ngspice, batch mode, `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |

## Status

| Part | State |
|---|---|
| A — datasets (NMOS ON / OFF, PMOS ON / OFF, 20,000 rows each) | done, all data checks pass — [`Data_Generation/README.md`](Data_Generation/README.md) |
| A — data check (Kirchhoff check, three trend plots) | done — `Data_Generation/Data_Check/` |
| A — four transistor models + model comparison | done — [`ML_Models/README.md`](ML_Models/README.md) |
| A — NMOS_OFF error analysis, structured (width / temperature) splits, Huber-loss models | done — [`Report/error_analysis_NMOS_OFF.md`](Report/error_analysis_NMOS_OFF.md), follow-up study in [`ML_Models/README.md`](ML_Models/README.md) |
| B — gate-level validation (INV, NAND2, NOR2, AND2, OR2; 18 cases × 200 corners) | not started |
| C — 3-bit CLA adder validation | not started |
| Example netlists as files (4 transistor, 5 gate, 1 circuit) | not yet saved as files (the transistor netlists are generated inside the simulation scripts) |
| Report | not started |

## Transistor-model results (test set, average over id, ig, is, ib)

| Case | Chosen model | Median error | 95th pct error | Mean error |
|---|---|---|---|---|
| NMOS_ON | MLP | 0.22 % | 1.27 % | 0.77 % |
| NMOS_OFF | MLP + Kirchhoff loss | 0.40 % | 2.60 % | 6.67 % (a few near-zero-crossing rows dominate it; 0.92 % if the 5 worst of 2,000 rows of each current are excluded) |
| PMOS_ON | MLP | 0.35 % | 1.56 % | 0.65 % |
| PMOS_OFF | MLP + Kirchhoff loss | 0.36 % | 2.13 % | 1.30 % |

Details, the 4 × 5 model comparison, per-current results and the error analysis are in
[`ML_Models/README.md`](ML_Models/README.md).

**With a Huber loss instead of MSE** (alternative models in `ML_Models/Saved_Models/huber/`, the official models above are unchanged) the test median / 95th percentile / ampere-weighted errors are NMOS_ON 0.10 / 0.62 / 0.08 %; NMOS_OFF 0.13 / 0.99 / 0.35 %; PMOS_ON 0.09 / 0.60 / 0.08 %; PMOS_OFF 0.12 / 0.86 / 0.25 % (official: 0.22 / 1.27 / 0.19 %; 0.40 / 2.60 / 0.76 %; 0.35 / 1.56 / 0.36 %; 0.36 / 2.13 / 0.71 %). The NMOS_OFF mean error (5.73 %) is still set by one near-cancelled row; see the [report section](Report/error_analysis_NMOS_OFF.md).

## Repository layout

```
45nm_MGK.pm                         SPICE model file
DVD_ML_PROJECT.pdf                  assignment
requirements.txt                    package versions used
Data_Generation/
  Datasets/                         *_20k.csv (training data) and the original 10k CSVs
  Scripts/                          sample generation + ngspice simulation
  Data_Check/                       data_check.ipynb, Kirchhoff check, trend plots
ML_Models/
  Notebooks/                        pmos_models.ipynb, nmos_models.ipynb (training)
  Results/                          model_evaluation.ipynb, validation / test tables
  Plots/                            predicted-vs-SPICE plots
  Saved_Models/                     NMOS_ON.pt, NMOS_OFF.pt, PMOS_ON.pt, PMOS_OFF.pt (official); huber/ = Huber-loss versions
  Scripts/leakage_model.py          loader + feature code shared by the notebooks and Parts B / C
  Scripts/offstudy.py, run_offstudy.py, export_huber_models.py   follow-up study (Huber loss, structured splits)
Report/                             report sections (error analysis of NMOS_OFF)
Model Files-20261105/, Project 1 - Leakage Estimation Files-20261105/
                                    course-provided material (model archives, example netlist, reference PDF)
```

## Reproducing Part A

From the repository root, in this order (details in the two sub-READMEs):

1. `python3 Data_Generation/Scripts/generate_nmos_20k.py` and `generate_pmos_20k.py` — the 20,000-sample datasets
2. `Data_Generation/Data_Check/data_check.ipynb` — data checks
3. `ML_Models/Notebooks/pmos_models.ipynb`, `nmos_models.ipynb` — train and select the models
4. `ML_Models/Results/model_evaluation.ipynb` — test-set results and plots

Install the Python packages with `pip install -r requirements.txt` (CPU-only PyTorch: see the comments in that file) and
make sure `ngspice` is on the path.
