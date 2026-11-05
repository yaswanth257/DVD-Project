# plan.md — Full Project Execution Plan

**Project:** ML-Driven PVT Variation Aware Leakage Estimation  
**Course:** Digital VLSI Design (DVD), Sem 5  
**Last Updated:** 2026-11-05

---

## Overview

```
Part A  →  Data Generation + ML Training  (transistor level)
Part B  →  Gate-Level Validation          (INV, NAND2, NOR2, AND2, OR2)
Part C  →  Test Circuit Validation        (3-bit CLA or 8-bit Parity Checker)
```

---

## Part A — Transistor Models

### A1 · Data Generation

**Goal:** 4 CSV files × 10,000 rows, 16 columns each.

#### A1.1 — NMOS ON
- Script: `generate_nmos_on_inputs.py` → `simulate_nmos_on.py`
- Bias: `vg = VDD`, `vb = 0 V`
- Output: `Data_Generation/Datasets/NMOS_ON.csv`

#### A1.2 — NMOS OFF
- Script: `generate_nmos_off_inputs.py` → `simulate_nmos_off.py`
- Bias: `vg = 0 V`, `vb = 0 V`
- Output: `Data_Generation/Datasets/NMOS_OFF.csv`

#### A1.3 — PMOS ON
- Script: `generate_pmos_on_inputs.py` → `simulate_pmos_on.py`
- Bias: `vg = 0 V`, `vb = VDD`
- Output: `Data_Generation/Datasets/PMOS_ON.csv`

#### A1.4 — PMOS OFF
- Script: `generate_pmos_off_inputs.py` → `simulate_pmos_off.py`
- Bias: `vg = VDD`, `vb = VDD`
- Output: `Data_Generation/Datasets/PMOS_OFF.csv`

#### A1.5 — Data Quality Check
For each dataset verify:
- [ ] Shape = (10000, 16), zero nulls
- [ ] KCL: `|id + ig + is + ib|` ≈ 0 for all rows
- [ ] All column ranges within spec
- [ ] Trend plot 1: OFF current vs Temperature
- [ ] Trend plot 2: OFF current vs `vth0`
- [ ] Trend plot 3: Gate current vs `toxe`

---

### A2 · ML Model Training

**Goal:** Train and compare 4 ML model types on each of the 4 datasets (16 models total). Select best per dataset.

#### A2.1 — Data Preprocessing
- [ ] Load all 4 CSVs
- [ ] Consider log-transform targets: predict `log10(|id|), log10(|ig|), log10(|is|), log10(|ib|)` — accounts for many-decade current range
- [ ] Train/Val/Test split: **70/15/15** using fixed seed `12175305`
- [ ] Normalize / standardize features as needed per model type

#### A2.2 — Model Types to Compare (per dataset)
1. **Linear Regression** — baseline, interpretable
2. **Random Forest** — handles non-linearity, robust
3. **Gradient Boosting** — XGBoost or LightGBM, high accuracy
4. **Neural Network (MLP)** — flexible, can capture complex I-V curves

#### A2.3 — Training & Selection
- [ ] Train all 4 types on each of 4 datasets
- [ ] Evaluate on **validation set**: R², median error, 95th-percentile error
- [ ] Select best model per dataset based on validation metrics
- [ ] Report final performance on **test set**
- [ ] Generate predicted-vs-SPICE scatter plot for each of the 4 cases

#### A2.4 — Save Models
- [ ] Save each winning model to file (e.g. `.pkl` for sklearn, `.json` for XGBoost, `.pt` for PyTorch)
- [ ] Models must be loadable by Part B scripts without re-training

---

## Part B — Gate-Level Validation

**Goal:** Use trained transistor models to predict leakage of 5 CMOS gates across all input combinations, validate against SPICE.

### B1 · SPICE Reference Simulations

For each of the **18 gate-input cases**, simulate the full transistor-level netlist at **200 random corners**:

| Gate | Input combos | Corner draw |
|------|-------------|-------------|
| INV | 2 | NMOS draw + PMOS draw |
| NAND2 | 4 | shared per transistor type |
| NOR2 | 4 | shared per transistor type |
| AND2 | 4 | shared per transistor type |
| OR2 | 4 | shared per transistor type |

**Record from each SPICE run:**
- Total gate leakage current
- Internal node voltages (x for NAND2/AND2, y for NOR2/OR2)
- Output voltage

> These results are **reference only** — not used for training.

### B2 · Build Gate Netlists
- [ ] INV — 1 PMOS (pull-up) + 1 NMOS (pull-down)
- [ ] NAND2 — 2 PMOS parallel + 2 NMOS series
- [ ] NOR2 — 2 PMOS series + 2 NMOS parallel
- [ ] AND2 — NAND2 + INV
- [ ] OR2 — NOR2 + INV

Each netlist must include `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`.

### B3 · ML Leakage Prediction Pipeline

For the same 200 corners and 18 cases:
1. Use SPICE-recorded **internal node voltages + output voltage** to determine each transistor's terminal voltages (Vd, Vg, Vs, Vb)
2. Feed (VDD, Temp, W, Vd, Vg, Vs, Vb, process params) into corresponding trained transistor ML model
3. Sum predicted terminal currents across all transistors → gate leakage prediction

### B4 · Gate Results Reporting
- [ ] Gate results table: SPICE median, ML median, bias, median error, 95th-pct error — for all 18 cases
- [ ] Gate parity plot: ML leakage vs SPICE leakage on **log-log axes**, all 18×200 = 3600 points
- [ ] Comment on best / worst cases (which gate, which input combo, which corner)

---

## Part C — Test Circuit Validation

**Goal:** Compose full circuit from gate models, compare hierarchical ML prediction vs SPICE.

### C1 · SPICE Reference
- [ ] Build complete transistor-level netlist of assigned test circuit
- [ ] Simulate all input combinations × 200 corners (same corners as Part B)
- [ ] Record total circuit leakage

### C2 · ML Prediction (Hierarchical)
1. For each input vector → determine logic value at each gate input
2. Call corresponding gate model (from Part B)
3. Sum all gate leakages → circuit leakage prediction

> Assumption: every gate input is exactly 0 or VDD; inter-gate loading ignored.

### C3 · Test Circuit Results
- [ ] Results table: SPICE vs ML (same metrics as gate table)
- [ ] Parity plot (log-log)
- [ ] Compare accuracy vs gate-level

---

## Final Report Outline

1. **Setup** — model file, VDD, test circuit, simulator version
2. **Data Generation** — sampling method, netlist structure, train/val/test split, KCL check results
3. **Models** — 4 types tried, hyperparameters, preprocessing (log|I|?), comparison table, reason for final choice
4. **Gate Results** — table + parity plot, best/worst analysis
5. **Test-Circuit Results** — metrics + parity plot, comparison with gate accuracy
6. **Error Analysis** — which case/gate/corner has largest errors, and why
7. **Limitations & Improvements** — what constrains accuracy, what would improve it

---

## Key Constraints & Rules

- Use **only** `45nm_MGK.pm` — no other model files
- Gate inputs applied at exactly **0 V or VDD** (no intermediate logic levels)
- SPICE gate results **must not** be used for training (reference only)
- All NMOS in a gate share one process corner draw; all PMOS share another
- AND2 and OR2 simulated as **complete gates** in SPICE (not as separate NAND2 + INV)
- Fixed seed `12175305` everywhere for reproducibility

