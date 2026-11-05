# memory.md — Project Brain / Reference Sheet

**Project:** ML-Driven PVT Variation Aware Leakage Estimation  
**Course:** Digital VLSI Design (DVD), Sem 5  
**Author:** Sangameshwar  
**Last Updated:** 2026-11-05 22:06 IST

---

## 1. Assignment At a Glance

| Item | Value |
|------|-------|
| Model file | `45nm_MGK.pm` — PTM 45 nm Metal Gate / High-K, BSIM4 Level 54 |
| Nominal VDD | **0.9 V** |
| VDD range | 0.81 V – 0.99 V (±10 %) |
| Channel length L | **45 nm** (fixed for all transistors) |
| Width W | 45 nm – 360 nm (1× – 8× L_min), uniform random |
| Temperature | −40 °C to 125 °C, uniform random |
| Random seed | **`12175305`** — combined last two digits of team roll numbers (`...05`) |
| Test circuit | 3-bit CLA **or** 8-bit Parity Checker (confirm with TAs) |
| Simulator | ngspice-42, batch mode (`ngspice -b`) |
| Target accuracy | ≥ 99 % mean |

---

## 1.5 · Random Seed — Important

```python
SEED = 12175305
```

| Detail | Value |
|--------|-------|
| Seed value | **`12175305`** |
| How derived | Combined **last two digits** of team members' roll numbers → `...05` appended |
| Used in | Every script: `np.random.default_rng(12175305)` |
| Applies to | All 4 data generation scripts, train/val/test split, gate corner sampling |

> **Never change this seed.** All datasets, splits, and corner draws must use `12175305` to remain reproducible and consistent across the team.

---

## 2. Nominal Process Parameters

### NMOS
| Param | Nominal | ±10 % range |
|-------|---------|------------|
| `vth0` | +0.3423 V | 0.3081 – 0.3765 V |
| `toxe` | 9.0e-10 m | 8.1e-10 – 9.9e-10 m |
| `ndep` | 6.5e18 cm⁻³ | 5.85e18 – 7.15e18 |
| `u0` | 0.02947 m²/V·s | 0.02652 – 0.03242 |
| `xj` | 1.4e-8 m | 1.26e-8 – 1.54e-8 m |

### PMOS
| Param | Nominal | ±10 % range |
|-------|---------|------------|
| `vth0` | **−0.23122 V** | −0.2544 – −0.2081 V |
| `toxe` | 9.2e-10 m | 8.28e-10 – 1.012e-9 m |
| `ndep` | 2.8e18 cm⁻³ | 2.52e18 – 3.08e18 |
| `u0` | 0.00391 m²/V·s | 0.003519 – 0.004301 |
| `xj` | 1.4e-8 m | 1.26e-8 – 1.54e-8 m |

---

## 3. Bias Conditions (Spec-Defined)

| Case | `vg` | `vb` |
|------|------|------|
| NMOS ON  | VDD | 0 V  |
| NMOS OFF | 0 V | 0 V  |
| PMOS ON  | 0 V | VDD  |
| PMOS OFF | VDD | VDD  |

---

## 4. Dataset Schema (all 4 CSV files)

```
vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib
```
16 columns × 10,000 rows per file.  
Currents `id, ig, is, ib` come from ngspice DC OP simulation.

---

## 5. KCL Residuals (Quality Gate)

| Dataset | Max \|id+ig+is+ib\| | Status |
|---------|----------------------|--------|
| NMOS_ON  | (check NMOS run log) | ✅ Done |
| NMOS_OFF | (check NMOS run log) | ✅ Done |
| PMOS_ON  | **9.69 × 10⁻¹⁰ A** (~1 nA) | ✅ Done |
| PMOS_OFF | **9.10 × 10⁻¹³ A** (~1 pA) | ✅ Done |

---

## 6. Gate Definitions (Part B)

| Gate | Inputs | Internal node(s) |
|------|--------|-----------------|
| INV   | A ∈ {0,1}         | None |
| NAND2 | AB ∈ {00,01,10,11} | x (between series NMOS) |
| NOR2  | AB ∈ {00,01,10,11} | y (between series PMOS) |
| AND2  | AB ∈ {00,01,10,11} | x and NAND2 output node |
| OR2   | AB ∈ {00,01,10,11} | y and NOR2 output node |

- **18 gate-input cases** total  
- Each case simulated at **200 random corners**  
- Corner = 1 draw of (VDD, Temp, NMOS process params, PMOS process params)  
- All NMOS in a gate share one draw; all PMOS share another  

---

## 7. ML Model Plan

- **Split:** 70 / 15 / 15 % (train / val / test) — fixed seed  
- **4 datasets × 4 ML model types** = 16 models to compare  
- Select best on validation set → report test set performance  
- Metric targets: R², median error, 95th-percentile error  
- Consider predicting **log|I|** instead of I (leakage spans many decades)  

Candidate model types (to try all 4):
1. Linear Regression (baseline)  
2. Random Forest  
3. Gradient Boosting (XGBoost / LightGBM)  
4. Neural Network (MLP)  

---

## 8. Deliverables Checklist (from spec)

1. 4 × CSV datasets (NMOS ON/OFF, PMOS ON/OFF)  
2. Example netlists (4 transistor + 5 gate + 1 test circuit)  
3. Data check: KCL residuals + 3 trend plots  
4. Model comparison table (4 types × 4 cases)  
5. Chosen models: test metrics + predicted-vs-SPICE plots + saved model files  
6. Gate results table (18 cases)  
7. Gate parity plot (log-log, ML vs SPICE)  
8. Test-circuit results table + parity plot  
9. All code (reproducible, fixed seeds)  
10. Typed report  

---

## 9. File & Folder Map

```
DVD-Project/
├── 45nm_MGK.pm                          ← SPICE model file
├── memory.md                            ← This file
├── plan.md                              ← Full project plan
├── status.md                            ← Live progress tracker
├── README_sangameshwar_pmos_dataset.md  ← PMOS dataset documentation
├── DVD_ML_PROJECT.pdf                   ← Full project spec
├── Project-1_leakage_estimation_annotated.pdf  ← Annotated leakage reference
├── Data_Generation/
│   ├── Scripts/
│   │   ├── generate_nmos_on_inputs.py
│   │   ├── generate_nmos_off_inputs.py
│   │   ├── simulate_nmos_on.py
│   │   ├── simulate_nmos_off.py
│   │   ├── generate_pmos_on_inputs.py   ← NEW
│   │   ├── generate_pmos_off_inputs.py  ← NEW
│   │   ├── simulate_pmos_on.py          ← NEW
│   │   ├── simulate_pmos_off_inputs.py  ← NEW
│   │   └── generate_pmos_datasets.py    ← NEW (standalone runner)
│   └── Datasets/
│       ├── NMOS_ON.csv   (10,000 rows, 16 cols) ✅
│       ├── NMOS_OFF.csv  (10,000 rows, 16 cols) ✅
│       ├── PMOS_ON.csv   (10,000 rows, 16 cols) ✅ NEW
│       └── PMOS_OFF.csv  (10,000 rows, 16 cols) ✅ NEW
├── Model Files-20261105/
│   └── models.zip
└── Project 1 - Leakage Estimation Files-20261105/
    ├── 22n_NMOS_ON.net
    ├── model.tar
    └── Project-1_leakage_estimation_annotated.pdf
```

---

## 10. Environment Notes

| Tool | Version / Path |
|------|---------------|
| Python | 3.12.3 — `/usr/bin/python3` |
| numpy | 1.26.4 — `/usr/lib/python3/dist-packages` |
| pandas | ❌ Not system-available — use `csv` stdlib instead |
| ngspice | 42 — `/usr/bin/ngspice` |
| SPICE options | `gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |
| ngspice param override | `altermod <type> <param> = <value>` |
| Current extraction | `echo RESULT $&id_val $&ig_val $&is_val $&ib_val` |

