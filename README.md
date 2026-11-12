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
| Random seed | 12175305 (every script and notebook) |
| Samples per transistor case | 20,000 (80 / 10 / 10 split → 16,000 / 2,000 / 2,000) |
| Simulator | ngspice, batch mode, `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |

## Status

| Part | State |
|---|---|
| A — datasets (NMOS ON / OFF, PMOS ON / OFF, 20,000 rows each) | done, all data checks pass — [`Data_Generation/README.md`](Data_Generation/README.md) |
| A — data check (Kirchhoff check, three trend plots) | done — `Data_Generation/Data_Check/` |
| A — model comparison + four final transistor models (MLP, Huber loss; all four also trained on gate-region rows) | done — [`ML_Models/README.md`](ML_Models/README.md) |
| A — NMOS_OFF error analysis, structured (width / temperature) splits, Huber loss | done — [`Report/error_analysis_NMOS_OFF.md`](Report/error_analysis_NMOS_OFF.md), follow-up study in [`ML_Models/README.md`](ML_Models/README.md) |
| A — Huber models, 4 seeds per case re-trained on the IIIT Turing cluster | done — [`ML_Models/Results/cluster_huber/`](ML_Models/Results/cluster_huber/) |
| B — gate-level validation (INV, NAND2, NOR2, AND2, OR2; 18 cases × 200 corners) | done — median error 0.17 %, mean accuracy 99.77 % — [`Gate_Validation/README.md`](Gate_Validation/README.md) |
| Example netlists (4 transistor, 5 gate) | done — [`Data_Generation/Netlists/`](Data_Generation/Netlists/), [`Gate_Validation/Netlists/`](Gate_Validation/Netlists/) |
| Report | not started |

## Deliverables (assignment section 6)

| # | Deliverable | Where | State |
|---|---|---|---|
| 1 | Four datasets, 10,000 rows, spec column order | `Data_Generation/Datasets/{NMOS,PMOS}_{ON,OFF}.csv` (training uses the 20,000-row `*_20k.csv`, whose first 10,000 rows are these files) | done |
| 2 | Example netlists | `Data_Generation/Netlists/*.cir` (4 transistor cases), `Gate_Validation/Netlists/*.cir` (5 gates) | done |
| 3 | Data check: Kirchhoff check + 3 trend plots | `Data_Generation/Data_Check/` (`kirchhoff_check.csv`, `fig1`–`fig4`) | done |
| 4 | Model comparison on the validation set (R², median, 95th) | `ML_Models/Results/validation_comparison_{NMOS,PMOS,HUBER}.csv` (random forest, XGBoost, polynomial ridge, MLP, MLP + Kirchhoff loss, MLP with Huber loss) and `validation_comparison_GATE_REGION.csv` (final models), table in `ML_Models/Results/model_evaluation.ipynb` | done |
| 5 | Chosen models: test median / 95th, parity plots, model files | `ML_Models/Results/test_results.csv`, `ML_Models/Plots/parity_*.png`, `ML_Models/Saved_Models/*.pt` (the MLP with Huber loss, also trained on gate-region rows) | done |
| 6 | Gate results table (18 cases: SPICE median, ML median, bias, median and 95th-percentile error) | `Gate_Validation/Results/gate_results_table_{total,vdd}.csv` | done |
| 7 | Gate parity plot, log axes | `Gate_Validation/Plots/gate_parity_{total,vdd}.png` | done |
| 9 | Code that regenerates everything, fixed seeds | [Reproducing Part A](#reproducing-part-a); Part B: `Gate_Validation/Scripts/01` – `06` | Parts A and B done |
| 10 | Report | — | not started |

## Transistor-model results (test set, average over id, ig, is, ib)

Final models: one MLP per case (25 inputs → 128-64-32-16 → 4 outputs) trained with a Huber loss, `ML_Models/Saved_Models/<case>.pt`;
all four models were also trained on 4,000 extra rows from the region where the transistor operates inside gates (for Part B).

| Case | R² (log\|I\|) | Median error | 95th pct error | Mean error | Ampere-weighted error |
|---|---|---|---|---|---|
| NMOS_ON | 0.9997 | 0.15 % | 0.88 % | 0.63 % | 0.14 % |
| NMOS_OFF | 0.9887 | 0.14 % | 1.30 % | 3.07 % ¹ | 0.29 % |
| PMOS_ON | 0.9997 | 0.14 % | 0.79 % | 0.41 % | 0.13 % |
| PMOS_OFF | 0.9959 | 0.14 % | 1.15 % | 1.13 % | 0.29 % |

¹ Set by a handful of near-zero-crossing rows: one test row contributes 7.4 points of the `is` mean; without the 5 worst of
2,000 rows the `id` / `is` means are 0.61 % / 0.51 %. The ampere-weighted error is Σ|I_pred − I_true| / Σ|I_true|.

Details — the comparison of the model variants on the validation set, per-current results, why the Huber loss, why the
gate-region rows and the error analysis — are in [`ML_Models/README.md`](ML_Models/README.md). The Huber models were also
re-trained with 4 seeds per case on the Turing cluster: test median 0.09 – 0.14 %, 95th percentile 0.60 – 1.22 % in every case and
seed ([`ML_Models/Results/cluster_huber/`](ML_Models/Results/cluster_huber/)); no extra seed was better on the validation set.

## Gate-level results (Part B, 18 cases × 200 corners = 3,600 operating points)

| Leakage | Bias | Median error | 95th pct error | Mean accuracy |
|---|---:|---:|---:|---:|
| total (VDD source + inputs at logic 1) | −0.04 % | 0.17 % | 0.63 % | 99.77 % |
| VDD source only | −0.03 % | 0.18 % | 0.85 % | 99.71 % |

Every gate / input case has a median error ≤ 0.35 % (total leakage). Per-case table, method and parity plots:
[`Gate_Validation/README.md`](Gate_Validation/README.md).

## Part A in brief

* **Data.** For each case, 20,000 single-transistor ngspice operating points with every parameter drawn as in section 4 of the
  assignment (seed 12175305); rows 1 – 10,000 are the 10k deliverable datasets. Every check passes: spec ranges, column order,
  no duplicates or zero currents, and Kirchhoff (`|id + ig + is + ib|` at most 2.2e-4 of the largest current of a sample).
* **Models.** Five model types were compared on the validation set (random forest, XGBoost, polynomial ridge, MLP, MLP with a
  Kirchhoff loss term, all with the MSE loss), predicting `log10|I|` of the four currents from 10 raw inputs and 15 physics
  features. The MLP is about 10× better in median error than the next model type.
* **Final models: the MLP with a Huber loss.** A follow-up study trained the same MLP with a Huber loss instead of MSE, so that
  the few near-cancelled rows no longer dominate the fit. It is the best variant on the validation median and 95th percentile in
  all four cases and lowers them 2.3 – 3.7× and 2.0 – 2.9× against the MSE MLP. It is also stable across training seeds (4 seeds
  per case re-trained on the Turing cluster).
* **All models: + gate-region rows.** Inside a gate an ON transistor has both terminals near one rail and |Vds| of 20 µV – 1 mV,
  where uniform sampling puts almost no data; Part B showed the ON models' gate and body currents were 2 – 7 % off there. 4,000
  extra single-transistor rows per ON case from that region (own random draws, no gate result) fixed it, at a cost of about 0.05
  points of median error on the uniform test rows. A gate-level diagnostic then showed that the remaining error sat in the channel
  current of OFF transistors at |Vds| > 0.1 V: gates put them with drain and source at the rails (52 % at opposite rails), where
  the 20,000 uniform rows have about 8 rows within 2 % of VDD of that corner. 4,000 extra rows per OFF case from those cells halved
  the gate-level error (median 0.34 % → 0.17 %) at a cost of 0.01 – 0.02 points of median and about 0.3 points of 95th percentile error on
  the uniform test rows. Final test median 0.14 – 0.15 %, 95th percentile 0.79 – 1.30 %. These rows are not drawn from the section 4
  distribution and were chosen after looking at the gate-level results; the report should say so.
* **NMOS_OFF mean error (3.07 %; 5.73 % before the gate-region rows).** It comes from a handful of samples where one terminal current nearly cancels (down to about
  1e-6 of the sample's largest current), so a tiny absolute error is a huge relative one; at the worst sample SPICE's own
  Kirchhoff residual is 2 % of that current. Seeds, larger networks and the loss do not remove it; details in
  [`Report/error_analysis_NMOS_OFF.md`](Report/error_analysis_NMOS_OFF.md).
* **Current signs.** The models predict magnitudes; `LeakageModel.predict_signed` adds the sign with a rule that holds on every
  dataset row (99.1 – 100 % of test rows with predicted magnitudes) — needed to sum currents at a node in Part B.

## Part B in brief

* **Setup** (`Gate_Validation/`): 200 PVT corners (one NMOS and one PMOS process draw per corner), INV / NAND2 / NOR2 / AND2 / OR2
  with every W in 45 – 360 nm, inputs exactly 0 or VDD, AND2 / OR2 simulated as complete gates; leakage reported both as *total*
  (VDD source + input sources at logic 1) and *VDD only*, with the same definition in SPICE and ML.
* **ML prediction:** the SPICE node voltages give each transistor's terminal voltages; the model is chosen by the gate voltage;
  `predict_signed` gives the signed currents. The channel current of every ON transistor is solved from Kirchhoff's current law
  at the internal nodes (`gate_common.compose_currents`): it is fixed by the rest of the gate and cannot be resolved by a model
  from a |Vds| of 0.1 mV. Leakage = −Σ currents at the VDD (and logic-1 input) terminals.
* **How the error came down:** first version (plain sum of the model currents) median 4.22 %, mean accuracy 94.6 %; with the
  Kirchhoff composition 1.42 % / 97.7 %; with the ON models trained on gate-region rows 0.34 % / 99.5 %; with the OFF models trained on
  gate-region rows too 0.17 % / 99.77 %.

## Repository layout

```
45nm_MGK.pm                         SPICE model file
DVD_ML_PROJECT.pdf                  assignment
requirements.txt                    package versions used
Data_Generation/
  Datasets/                         *_20k.csv (training data), the original 10k CSVs, *_gate_region.csv (extra training rows)
  Scripts/                          sample generation + ngspice simulation, example-netlist writer
  Netlists/                         example single-transistor netlists, one per case
  Data_Check/                       data_check.ipynb, Kirchhoff check, trend plots
ML_Models/
  Notebooks/                        pmos_models.ipynb, nmos_models.ipynb (comparison of the five model types)
  Results/                          model_evaluation.ipynb (deliverables 4, 5), validation / test tables
  Results/offstudy/                 runs of the follow-up study (metrics, predictions; read by nmos_off_study.ipynb)
  Results/cluster_huber/            4-seed Huber re-training on the cluster (summary tables)
  Plots/                            predicted-vs-SPICE plots
  Saved_Models/                     NMOS_ON.pt, NMOS_OFF.pt, PMOS_ON.pt, PMOS_OFF.pt — the final models
  Scripts/leakage_model.py          loader (predict, predict_signed) + feature code shared by the notebooks and Part B
  Scripts/check_sign_rule.py        checks the current-sign rule -> Results/sign_rule_check.csv
  Scripts/offstudy.py, run_offstudy.py, export_huber_models.py   follow-up study and the first Huber models
  Scripts/train_on_models_gate_region.py   training / installing the final models (ON and OFF, with gate-region rows)
  Results/gate_region/              training runs of the final models
Gate_Validation/                    Part B: Scripts/01 – 06, Netlists/, Results/ (deliverable 6), Plots/ (deliverable 7)
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
   `python3 ML_Models/Scripts/export_huber_models.py` (keeps the better seed on validation → the final OFF models), then
   `python3 Data_Generation/Scripts/generate_gate_region_data.py` and
   `python3 ML_Models/Scripts/train_on_models_gate_region.py --cases all` (the final models of all four cases, trained with the
   gate-region rows; `export_huber_models.py` and `--stage 13` produce the Huber models of the follow-up study)
7. `ML_Models/Results/model_evaluation.ipynb` — comparison table, test-set results and parity plots
8. `python3 ML_Models/Scripts/check_sign_rule.py` — checks the current-sign rule used by `predict_signed`

Install the Python packages with `pip install -r requirements.txt` (CPU-only PyTorch: see the comments in that file) and
make sure `ngspice` is on the path.

**Part B:** `python3 Gate_Validation/Scripts/01_generate_corners.py` … `06_diagnose_models_in_gates.py 12`, in order (details in
[`Gate_Validation/README.md`](Gate_Validation/README.md)).

