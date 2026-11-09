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
| A — model comparison + four final transistor models (MLP, Huber loss) | done — [`ML_Models/README.md`](ML_Models/README.md) |
| A — NMOS_OFF error analysis, structured (width / temperature) splits, Huber loss | done — [`Report/error_analysis_NMOS_OFF.md`](Report/error_analysis_NMOS_OFF.md), follow-up study in [`ML_Models/README.md`](ML_Models/README.md) |
| A — Huber models, 4 seeds per case re-trained on the IIIT Turing cluster | done — [`ML_Models/Results/cluster_huber/`](ML_Models/Results/cluster_huber/) |
| B — gate-level validation (INV, NAND2, NOR2, AND2, OR2; 18 cases × 200 corners) | not started |
| C — 3-bit CLA adder validation | not started |
| Example netlists (4 transistor, 5 gate, 1 circuit) | transistor netlists done — [`Data_Generation/Netlists/`](Data_Generation/Netlists/); gate and circuit netlists come with Parts B and C |
| Report | not started |

## Deliverables (assignment section 6)

| # | Deliverable | Where | State |
|---|---|---|---|
| 1 | Four datasets, 10,000 rows, spec column order | `Data_Generation/Datasets/{NMOS,PMOS}_{ON,OFF}.csv` (training uses the 20,000-row `*_20k.csv`, whose first 10,000 rows are these files) | done |
| 2 | Example netlists | `Data_Generation/Netlists/*.cir` (4 transistor cases) | transistor netlists done; gate + CLA netlists pending |
| 3 | Data check: Kirchhoff check + 3 trend plots | `Data_Generation/Data_Check/` (`kirchhoff_check.csv`, `fig1`–`fig4`) | done |
| 4 | Model comparison on the validation set (R², median, 95th) | `ML_Models/Results/validation_comparison_{NMOS,PMOS,HUBER}.csv` (random forest, XGBoost, polynomial ridge, MLP, MLP + Kirchhoff loss, MLP with Huber loss), table in `ML_Models/Results/model_evaluation.ipynb` | done |
| 5 | Chosen models: test median / 95th, parity plots, model files | `ML_Models/Results/test_results.csv`, `ML_Models/Plots/parity_*.png`, `ML_Models/Saved_Models/*.pt` (the MLP with Huber loss for all four cases) | done |
| 6 – 8 | Gate results table + parity plot, CLA results + parity plot | — | Parts B / C |
| 9 | Code that regenerates everything, fixed seeds | the scripts and notebooks listed in [Reproducing Part A](#reproducing-part-a) | Part A done |
| 10 | Report | — | not started |

## Transistor-model results (test set, average over id, ig, is, ib)

Final models: one MLP per case (25 inputs → 128-64-32-16 → 4 outputs) trained with a Huber loss, `ML_Models/Saved_Models/<case>.pt`.

| Case | R² (log\|I\|) | Median error | 95th pct error | Mean error | Ampere-weighted error |
|---|---|---|---|---|---|
| NMOS_ON | 0.9997 | 0.10 % | 0.62 % | 0.58 % | 0.08 % |
| NMOS_OFF | 0.9923 | 0.13 % | 0.99 % | 5.73 % ¹ | 0.35 % |
| PMOS_ON | 0.9998 | 0.09 % | 0.60 % | 0.31 % | 0.08 % |
| PMOS_OFF | 0.9964 | 0.12 % | 0.86 % | 0.85 % | 0.25 % |

¹ Set by a handful of near-zero-crossing rows: one test row contributes 19.5 points of the `is` mean; without the 5 worst of
2,000 rows the `id` / `is` means are 0.50 % / 0.43 %. The ampere-weighted error is Σ|I_pred − I_true| / Σ|I_true|.

Details — the comparison of six model variants on the validation set, per-current results, why the Huber loss and the error
analysis — are in [`ML_Models/README.md`](ML_Models/README.md). The Huber models were also re-trained with 4 seeds per case on
the Turing cluster: test median 0.09 – 0.14 %, 95th percentile 0.60 – 1.22 % in every case and seed
([`ML_Models/Results/cluster_huber/`](ML_Models/Results/cluster_huber/)); no extra seed was better on the validation set.

## Part A in brief

* **Data.** For each case, 20,000 single-transistor ngspice operating points with every parameter drawn as in section 4 of the
  assignment (seed 12175305); rows 1 – 10,000 are the 10k deliverable datasets. Every check passes: spec ranges, column order,
  no duplicates or zero currents, and Kirchhoff (`|id + ig + is + ib|` at most 2.2e-4 of the largest current of a sample).
* **Models.** Five model types were compared on the validation set (random forest, XGBoost, polynomial ridge, MLP, MLP with a
  Kirchhoff loss term, all with the MSE loss), predicting `log10|I|` of the four currents from 10 raw inputs and 15 physics
  features. The MLP is about 10× better in median error than the next model type.
* **Final models: the MLP with a Huber loss.** A follow-up study trained the same MLP with a Huber loss instead of MSE, so that
  the few near-cancelled rows no longer dominate the fit. It is the best variant on the validation median and 95th percentile in
  all four cases and lowers them 2.3 – 3.7× and 2.0 – 2.9× against the MSE MLP; test median 0.09 – 0.13 %, 95th percentile
  0.60 – 0.99 %. It is also stable across training seeds (4 seeds per case re-trained on the Turing cluster).
* **NMOS_OFF mean error (5.73 %).** It comes from a handful of samples where one terminal current nearly cancels (down to about
  1e-6 of the sample's largest current), so a tiny absolute error is a huge relative one; at the worst sample SPICE's own
  Kirchhoff residual is 2 % of that current. Seeds, larger networks and the loss do not remove it; details in
  [`Report/error_analysis_NMOS_OFF.md`](Report/error_analysis_NMOS_OFF.md).
* **Current signs.** The models predict magnitudes; `LeakageModel.predict_signed` adds the sign with a rule that holds on every
  dataset row (99.6 – 100 % of test rows with predicted magnitudes) — needed to sum currents at a node in Parts B and C.

## Starting Parts B and C

**What is ready**

| Need | Use |
|---|---|
| Transistor currents with sign | `LeakageModel(path).predict_signed(df)` in `ML_Models/Scripts/leakage_model.py` — one row per transistor, columns `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj`; returns `id, ig, is, ib` in A, + = current out of the transistor ([details](ML_Models/README.md#current-signs-predict_signed-for-parts-b-and-c)) |
| Model files | `ML_Models/Saved_Models/<case>.pt` (the four final models) |
| Netlist template | `Data_Generation/Netlists/*.cir`: model include, the required `.options` line, process parameters set with `altermod`, currents read with `i(V…)` |
| Corner draws | the section 4 distributions as coded in `Data_Generation/Scripts/generate_*_20k.py` (VDD ± 10 %, −40 … 125 °C, process ± 10 % around the NMOS / PMOS nominals) |

**Building the ML prediction of one gate case (assignment step 4)**

1. For each corner, draw VDD, temperature, one NMOS process set and one PMOS process set (all NMOS of the gate share one draw,
   all PMOS the other).
2. From the SPICE run, take the output voltage and the internal node voltages (`x` in NAND2 / AND2, `y` in NOR2 / OR2, and the
   NAND2 / NOR2 output inside AND2 / OR2).
3. For every transistor, build one input row: `vd`, `vs` from those node voltages (or 0 / VDD), `vg` = its input (exactly 0 or
   VDD), `vb` = 0 for NMOS and VDD for PMOS, `w` = its width, process values from the NMOS or PMOS draw.
4. Pick the model by the gate voltage: NMOS with `vg = VDD` → NMOS_ON, `vg = 0` → NMOS_OFF; PMOS with `vg = 0` → PMOS_ON,
   `vg = VDD` → PMOS_OFF. These are exactly the four training cases, and `vd`, `vs` ∈ [0, VDD] is inside the trained range.
5. Gate leakage = current drawn from the supply = −Σ of the signed currents of every terminal connected to VDD (PMOS
   sources / drains at VDD and every PMOS body). In ngspice the same quantity is `−i(Vdd)`.

**Decisions to make before coding**

* **Transistor widths in the gates** — any sizing, but every W must stay within 45 – 360 nm, the trained range.
* **What counts as gate leakage** — the VDD supply current only, or also the currents of the input sources (the gate current of
  transistors whose gate sits at VDD flows from the input source, not from VDD). SPICE and ML must use the same definition.
* **"Internal node voltages from both"** (section 5) — step 4 says to use the SPICE node voltages for the ML prediction; an
  ML estimate of `x` / `y` would mean solving Kirchhoff at that node with the models (e.g. bisection on its voltage). Worth
  confirming with the TAs.
* **The CLA must be built only from INV, NAND2, NOR2, AND2 and OR2**, because Part C composes the circuit from the gate models.
  With carry-in it has 7 inputs: 128 input vectors × 200 corners in SPICE.

**What to expect from the transistor models.** The near-cancelled current of an OFF transistor sits at the lower-voltage
terminal for NMOS and the higher-voltage terminal for PMOS; it is small in absolute terms, so it only matters if it is part of
the supply-current sum. Deliverable 6 asks per gate case for the SPICE median, ML median, bias, median error and 95th-percentile
error; deliverables 7 and 8 for log-log parity plots.

## Repository layout

```
45nm_MGK.pm                         SPICE model file
DVD_ML_PROJECT.pdf                  assignment
requirements.txt                    package versions used
Data_Generation/
  Datasets/                         *_20k.csv (training data) and the original 10k CSVs
  Scripts/                          sample generation + ngspice simulation, example-netlist writer
  Netlists/                         example single-transistor netlists, one per case
  Data_Check/                       data_check.ipynb, Kirchhoff check, trend plots
ML_Models/
  Notebooks/                        pmos_models.ipynb, nmos_models.ipynb (comparison of the five model types)
  Results/                          model_evaluation.ipynb (deliverables 4, 5), validation / test tables
  Results/offstudy/                 runs of the follow-up study (metrics, predictions; read by nmos_off_study.ipynb)
  Results/cluster_huber/            4-seed Huber re-training on the cluster (summary tables)
  Plots/                            predicted-vs-SPICE plots
  Saved_Models/                     NMOS_ON.pt, NMOS_OFF.pt, PMOS_ON.pt, PMOS_OFF.pt — the final models (MLP, Huber loss)
  Scripts/leakage_model.py          loader (predict, predict_signed) + feature code shared by the notebooks and Parts B / C
  Scripts/check_sign_rule.py        checks the current-sign rule -> Results/sign_rule_check.csv
  Scripts/offstudy.py, run_offstudy.py, export_huber_models.py   follow-up study and training / installing the final models
Report/                             report sections (error analysis of NMOS_OFF)
```

## Reproducing Part A

From the repository root, in this order (details in the two sub-READMEs):

1. The 10,000-sample datasets (deliverable 1): `python3 Data_Generation/Scripts/simulate_nmos_on.py`, `simulate_nmos_off.py`,
   `generate_pmos_datasets.py`
2. The 20,000-sample training datasets: `generate_nmos_20k.py`, `generate_pmos_20k.py` (rows 1 – 10,000 repeat step 1;
   `generate_nmos_20k.py` checks that and stops if they differ, so step 1 comes first)
3. `python3 Data_Generation/Scripts/write_example_netlists.py` — the four example netlists, checked in ngspice
4. `Data_Generation/Data_Check/data_check.ipynb` — data checks
5. `ML_Models/Notebooks/pmos_models.ipynb`, `nmos_models.ipynb` — the comparison of the five model types (MSE loss)
6. The final models: `python3 ML_Models/Scripts/run_offstudy.py --stage 13` (MLP with Huber loss, two seeds per case), then
   `python3 ML_Models/Scripts/export_huber_models.py` (keeps the better seed on validation → `ML_Models/Saved_Models/<case>.pt`)
7. `ML_Models/Results/model_evaluation.ipynb` — comparison table, test-set results and parity plots
8. `python3 ML_Models/Scripts/check_sign_rule.py` — checks the current-sign rule used by `predict_signed`

Install the Python packages with `pip install -r requirements.txt` (CPU-only PyTorch: see the comments in that file) and
make sure `ngspice` is on the path.
