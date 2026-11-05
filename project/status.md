# status.md — Live Progress Tracker

**Project:** ML-Driven PVT Variation Aware Leakage Estimation  
**Course:** Digital VLSI Design (DVD), Sem 5  
**Last Updated:** 2026-11-05 22:06 IST

> **Seed: `12175305`** — Combined last two digits of team roll numbers. Fixed across all scripts.

---

## Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Complete |
| 🔄 | In Progress |
| ⬜ | Not Started |
| ❌ | Blocked |

---

## Part A — Transistor Models

### A1 · Data Generation

| Task | Status | Notes |
|------|--------|-------|
| NMOS ON dataset (10,000 rows) | ✅ | `Data_Generation/Datasets/NMOS_ON.csv` — 2.7 MB |
| NMOS OFF dataset (10,000 rows) | ✅ | `Data_Generation/Datasets/NMOS_OFF.csv` — 2.6 MB |
| PMOS ON dataset (10,000 rows) | ✅ | `Data_Generation/Datasets/PMOS_ON.csv` — 2.7 MB · KCL max: 9.69e-10 A |
| PMOS OFF dataset (10,000 rows) | ✅ | `Data_Generation/Datasets/PMOS_OFF.csv` — 2.9 MB · KCL max: 9.10e-13 A |
| All 4 scripts written (PMOS) | ✅ | `generate_pmos_*.py`, `simulate_pmos_*.py` |
| Standalone PMOS runner | ✅ | `generate_pmos_datasets.py` (numpy only, no pandas) |
| README for PMOS dataset | ✅ | `README_sangameshwar_pmos_dataset.md` |

### A1-Check · Data Quality

| Check | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|-------|---------|---------|---------|---------|
| Shape (10000, 16) | ✅ | ✅ | ✅ | ✅ |
| Zero nulls | ✅ | ✅ | ✅ | ✅ |
| KCL residual | ✅ | ✅ | ✅ | ✅ |
| Column ranges valid | ✅ | ✅ | ✅ | ✅ |
| Trend plot: OFF current vs Temp | ⬜ | ⬜ | ⬜ | ⬜ |
| Trend plot: OFF current vs vth0 | ⬜ | ⬜ | ⬜ | ⬜ |
| Trend plot: Gate current vs toxe | ⬜ | ⬜ | ⬜ | ⬜ |

### A2 · ML Model Training

| Task | Status | Notes |
|------|--------|-------|
| Data preprocessing (log-transform, split) | ⬜ | |
| Linear Regression — NMOS ON | ⬜ | |
| Linear Regression — NMOS OFF | ⬜ | |
| Linear Regression — PMOS ON | ⬜ | |
| Linear Regression — PMOS OFF | ⬜ | |
| Random Forest — NMOS ON | ⬜ | |
| Random Forest — NMOS OFF | ⬜ | |
| Random Forest — PMOS ON | ⬜ | |
| Random Forest — PMOS OFF | ⬜ | |
| Gradient Boosting — NMOS ON | ⬜ | |
| Gradient Boosting — NMOS OFF | ⬜ | |
| Gradient Boosting — PMOS ON | ⬜ | |
| Gradient Boosting — PMOS OFF | ⬜ | |
| Neural Network (MLP) — NMOS ON | ⬜ | |
| Neural Network (MLP) — NMOS OFF | ⬜ | |
| Neural Network (MLP) — PMOS ON | ⬜ | |
| Neural Network (MLP) — PMOS OFF | ⬜ | |
| Model comparison table (4×4) | ⬜ | |
| Best model selection (val set) | ⬜ | |
| Test set evaluation + plots | ⬜ | |
| Save 4 winning model files | ⬜ | |

---

## Part B — Gate-Level Validation

| Task | Status | Notes |
|------|--------|-------|
| INV SPICE netlist | ⬜ | |
| NAND2 SPICE netlist | ⬜ | |
| NOR2 SPICE netlist | ⬜ | |
| AND2 SPICE netlist | ⬜ | |
| OR2 SPICE netlist | ⬜ | |
| SPICE simulations — 18 cases × 200 corners | ⬜ | |
| Record internal node voltages | ⬜ | |
| ML prediction pipeline for gates | ⬜ | |
| Gate results table (18 rows) | ⬜ | |
| Gate parity plot (log-log) | ⬜ | |

---

## Part C — Test Circuit Validation

| Task | Status | Notes |
|------|--------|-------|
| Confirm assigned test circuit with TAs | ⬜ | 3-bit CLA or 8-bit Parity? |
| Build test circuit transistor netlist | ⬜ | |
| SPICE simulations (all inputs × 200 corners) | ⬜ | |
| Hierarchical ML prediction | ⬜ | |
| Test circuit results table | ⬜ | |
| Test circuit parity plot (log-log) | ⬜ | |

---

## Deliverables

| # | Deliverable | Status |
|---|------------|--------|
| 1 | 4 × CSV datasets | ✅ |
| 2 | Example netlists (4 transistor + 5 gate + 1 test circuit) | ⬜ |
| 3 | KCL data check + 3 trend plots | 🔄 KCL ✅, plots ⬜ |
| 4 | Model comparison table (4×4) | ⬜ |
| 5 | Chosen models: metrics + plots + saved files | ⬜ |
| 6 | Gate results table (18 cases) | ⬜ |
| 7 | Gate parity plot (log-log) | ⬜ |
| 8 | Test-circuit results + parity plot | ⬜ |
| 9 | Reproducible code (all scripts, fixed seeds) | 🔄 Data gen ✅, rest ⬜ |
| 10 | Typed report | ⬜ |

---

## Overall Progress

```
Part A — Data Generation  ████████████████████  100% ✅
Part A — ML Training      ░░░░░░░░░░░░░░░░░░░░    0% ⬜
Part B — Gate Validation  ░░░░░░░░░░░░░░░░░░░░    0% ⬜
Part C — Test Circuit     ░░░░░░░░░░░░░░░░░░░░    0% ⬜
Report                    ░░░░░░░░░░░░░░░░░░░░    0% ⬜
─────────────────────────────────────────────────────
Total                     ████░░░░░░░░░░░░░░░░   ~20%
```

---

## Blockers / Open Questions

| # | Issue | Assigned To | Status |
|---|-------|------------|--------|
| 1 | Confirm test circuit (CLA or Parity?) with TAs | Sangameshwar | ⬜ Open |
| 2 | pandas not installed system-wide — ML training needs a venv or conda env with scikit-learn, xgboost, etc. | Sangameshwar | ⬜ Open |
| 3 | Decide log-transform strategy for targets (log\|I\| or raw I) | Sangameshwar | ⬜ Open |

---

## Session Log

| Date | Work Done |
|------|-----------|
| 2026-11-05 | Read all project docs (PDF spec + annotated leakage PDF + model file) |
| 2026-11-05 | Generated NMOS_ON.csv and NMOS_OFF.csv (pre-existing, confirmed valid) |
| 2026-11-05 | Wrote all 5 PMOS scripts (`generate_pmos_on/off_inputs.py`, `simulate_pmos_on/off.py`, `generate_pmos_datasets.py`) |
| 2026-11-05 | Ran ngspice simulations → generated PMOS_ON.csv (10,000 rows, KCL: 9.69e-10 A) |
| 2026-11-05 | Ran ngspice simulations → generated PMOS_OFF.csv (10,000 rows, KCL: 9.10e-13 A) |
| 2026-11-05 | Created `README_sangameshwar_pmos_dataset.md` |
| 2026-11-05 | Created `memory.md`, `plan.md`, `status.md` in `project/` folder |
| 2026-11-05 | Clarified seed `12175305` = combined last two digits of team roll numbers (`...05`) — updated all docs |

---

> **Next step:** Set up Python environment with pandas + scikit-learn + xgboost, then start A2 (ML training).

