# Part B — Gate-level validation (INV, NAND2, NOR2, AND2, OR2)

Gate leakage predicted from the four transistor models, compared with transistor-level ngspice simulations of the gates.
18 gate / input cases (INV: 2, NAND2, NOR2, AND2, OR2: 4 each) × 200 PVT corners = **3,600 operating points**.

**Result (total leakage, all 3,600 points): median error 0.34 %, 95th percentile 1.64 %, bias +0.03 %, mean accuracy 99.47 %.**

## Workflow

Run from the repository root, in order (seed `12175305`; `ngspice` must be on the path).

| Step | Script | Output |
|---|---|---|
| 1 | `01_generate_corners.py` | `Results/corners.csv` — 200 PVT corners (also used by Part C) |
| 2 | `02_write_gate_netlists.py` | `Netlists/{INV,NAND2,NOR2,AND2,OR2}.cir` — example gate netlists (deliverable 2) |
| 3 | `03_run_spice_gates.py` | `Results/gate_spice_results.csv` — SPICE leakage, internal-node and output voltages |
| 4 | `04_predict_ml_gates.py` | `Results/gate_ml_results.csv` — ML leakage from the four transistor models |
| 5 | `05_analyze_gates.py` | `Results/gate_results_table_{total,vdd}.csv` (deliverable 6), `Plots/gate_parity_{total,vdd}.png` (deliverable 7), `Results/gate_composition_comparison.csv`, `Plots/gate_error_vs_conditions.png` |
| 6 | `06_diagnose_models_in_gates.py 12` | `Results/diag_per_transistor.csv`, `Results/diag_error_by_vds.csv` — diagnostic on 12 corners |

Shared definitions (topology, sizing, corners, netlists, leakage composition) are in `Scripts/gate_common.py`.
`Scripts/numpy_model.py` runs the saved models without PyTorch (same predictions); with PyTorch installed the scripts use
`LeakageModel` from `ML_Models/Scripts/leakage_model.py`.

## Setup

- **Corners:** one corner = one draw of VDD (0.9 V ± 10 %), temperature (−40 … 125 °C), one NMOS process set and one PMOS process
  set (vth0, toxe, ndep, u0, xj, nominal ± 10 %); all NMOS of a gate share the NMOS draw, all PMOS the PMOS draw.
- **Sizing** (L = 45 nm, every W inside the trained 45 – 360 nm): INV Wn / Wp = 90 / 180 nm; NAND2 180 / 180 nm (series NMOS
  doubled); NOR2 90 / 360 nm (series PMOS doubled). AND2 / OR2 = NAND2 / NOR2 followed by the INV, simulated as one netlist.
- **Inputs** exactly 0 or VDD; NMOS bodies at 0, PMOS bodies at VDD. Every netlist has the required `.options` line.
- **Leakage definitions** (the same for SPICE and ML):
  - **total** = current supplied by the VDD source + by the input sources at logic 1 (= total supply power / VDD);
  - **vdd** = current supplied by the VDD source only.

## ML prediction (step 4)

1. The SPICE internal-node and output voltages give `vd`, `vs` of every transistor (clipped to [0, VDD]); `vg` is its input
   (0 or VDD) or the internal node `n` (inverter of AND2 / OR2).
2. The model is chosen by the gate voltage (NMOS: `vg = VDD` → NMOS_ON, else NMOS_OFF; PMOS: `vg = 0` → PMOS_ON, else PMOS_OFF);
   `LeakageModel.predict_signed` returns `id, ig, is, ib` with their sign (+ = current out of the transistor).
3. **Kirchhoff composition** (`gate_common.compose_currents`): the channel current of every ON transistor is solved from Kirchhoff's
   current law at the internal nodes; its `ig`, `ib` and all currents of OFF transistors are the model's. Inside a gate an ON
   transistor connects a node to a rail at a |Vds| of only 20 µV – 1 mV and carries the leakage of the OFF transistors through its
   channel. That current is fixed by the rest of the gate, while the model would have to resolve it from a 0.1 mV Vds (median
   error 8 – 93 % for its `id` / `is`). Taking it from Kirchhoff uses only predicted currents and the circuit's node equations.
4. Leakage = − Σ currents at terminals on the VDD net (− Σ at terminals on inputs at logic 1, for *total*).

The ON transistor models were also trained with 4,000 extra single-transistor rows per ON case from the gate operating region
(both terminals near a rail, |Vds| down to 0.1 µV; `Data_Generation/Scripts/generate_gate_region_data.py`, own random draws,
no gate result used). Uniform sampling of vd / vs (section 4) almost never produces such points, and there the ON models' `ig`
and `ib` — the currents the Kirchhoff composition does use — were 2 – 7 % off; now 0.2 – 0.4 %
(`ML_Models/README.md`, "Final models"). The gate SPICE results of step 3 are used for evaluation only.

## Results (deliverable 6)

| Leakage | SPICE median (A) | ML median (A) | Bias | Median error | 95th pct error | Mean accuracy |
|---|---:|---:|---:|---:|---:|---:|
| total | 1.552 × 10⁻⁷ | 1.552 × 10⁻⁷ | +0.03 % | 0.34 % | 1.64 % | 99.47 % |
| vdd | 1.201 × 10⁻⁷ | 1.198 × 10⁻⁷ | +0.07 % | 0.45 % | 2.41 % | 99.26 % |

Error = ML / SPICE − 1; bias = median signed error; mean accuracy = 100 % − mean |error|. Per gate and input, total leakage
(200 corners each; `Results/gate_results_table_total.csv`, the *vdd* table in `gate_results_table_vdd.csv`):

| Gate | Input | SPICE median (A) | ML median (A) | Bias | Median error | 95th pct error | Mean accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| INV | 0 | 6.669e-08 | 6.624e-08 | −0.12 % | 0.42 % | 1.68 % | 99.36 % |
| INV | 1 | 8.163e-08 | 8.198e-08 | +0.12 % | 0.54 % | 1.94 % | 99.29 % |
| NAND2 | 00 | 9.277e-08 | 9.311e-08 | −0.04 % | 0.21 % | 0.64 % | 99.75 % |
| NAND2 | 01 | 1.356e-07 | 1.351e-07 | +0.00 % | 0.32 % | 2.38 % | 99.37 % |
| NAND2 | 10 | 7.543e-08 | 7.524e-08 | +0.05 % | 0.32 % | 1.38 % | 99.50 % |
| NAND2 | 11 | 2.136e-07 | 2.140e-07 | +0.13 % | 0.42 % | 1.75 % | 99.39 % |
| NOR2 | 00 | 2.100e-07 | 2.081e-07 | −0.18 % | 0.35 % | 1.37 % | 99.48 % |
| NOR2 | 01 | 2.443e-07 | 2.438e-07 | +0.03 % | 0.32 % | 1.88 % | 99.49 % |
| NOR2 | 10 | 1.141e-07 | 1.139e-07 | +0.03 % | 0.36 % | 1.38 % | 99.49 % |
| NOR2 | 11 | 7.791e-08 | 7.859e-08 | +0.26 % | 0.36 % | 2.82 % | 99.18 % |
| AND2 | 00 | 1.977e-07 | 1.970e-07 | +0.06 % | 0.25 % | 1.16 % | 99.62 % |
| AND2 | 01 | 2.197e-07 | 2.205e-07 | +0.03 % | 0.34 % | 1.82 % | 99.47 % |
| AND2 | 10 | 1.548e-07 | 1.552e-07 | +0.08 % | 0.38 % | 1.40 % | 99.50 % |
| AND2 | 11 | 2.791e-07 | 2.797e-07 | +0.13 % | 0.38 % | 1.52 % | 99.47 % |
| OR2 | 00 | 3.135e-07 | 3.123e-07 | −0.12 % | 0.31 % | 1.18 % | 99.58 % |
| OR2 | 01 | 3.124e-07 | 3.124e-07 | +0.01 % | 0.36 % | 1.60 % | 99.51 % |
| OR2 | 10 | 1.775e-07 | 1.778e-07 | −0.00 % | 0.37 % | 1.38 % | 99.51 % |
| OR2 | 11 | 1.452e-07 | 1.472e-07 | +0.07 % | 0.26 % | 2.38 % | 99.43 % |

Most accurate: NAND2 `00` (0.21 %), AND2 `00`, OR2 `11`; least accurate: INV `1` (0.54 %) and the 95th percentile of NOR2 `11`
(2.8 %). On the *vdd* definition NOR2 `11` is the weakest case (median 1.9 %): its VDD current is only 1.3e-8 A, the leakage
through two series OFF PMOS (about 10× below the median case), so the same absolute model error is a larger relative one.

## How the error was brought down

| Total leakage, 3,600 points | Bias | Median error | 95th pct | Mean accuracy |
|---|---:|---:|---:|---:|
| Plain sum of the model currents, earlier ON models (first version of Part B) | −3.31 % | 4.22 % | 13.87 % | 94.59 % |
| + Kirchhoff for the ON channels | +1.33 % | 1.42 % | 7.29 % | 97.71 % |
| + ON models trained with gate-region rows (final) | **+0.03 %** | **0.34 %** | **1.64 %** | **99.47 %** |
| (plain sum with the final models, for comparison) | −0.32 % | 3.20 % | 17.98 % | 94.61 % |

- **Step 6 shows where the error came from.** Summing standalone single-transistor SPICE currents, at the voltages each transistor
  sees inside the gate, reproduces the gate SPICE leakage to a median of 0.035 % (the composition and the signs are right). The
  model errors inside gates are small for |Vds| > 0.1 V (OFF: 0.4 – 1 %) and large only for ON transistors at |Vds| < 1 mV.
- **The first version's errors followed one pattern:** they were large exactly when an ON transistor sits on the VDD net (INV `0`,
  NAND2 `00`, NOR2 `00` / `01`: 8 – 12 %) and small otherwise (≤ 1.8 %), because the supply current then flows through that ON
  channel.
- **After Kirchhoff, the ON transistors' `ig` / `ib` were the remaining source:** replacing only those with single-transistor SPICE
  values (12 corners) brought the median from 1.74 % to 0.27 %. The gate-region rows fixed exactly that (median `ig` / `ib` error at
  |Vds| < 1 mV: NMOS 1.8 / 2.4 % → 0.19 / 0.27 %, PMOS 4.4 / 6.6 % → 0.35 / 0.29 %).

## Files

| Content | File |
|---|---|
| PVT corners | `Results/corners.csv` |
| SPICE gate results (leakage, node voltages) | `Results/gate_spice_results.csv` |
| ML predictions (Kirchhoff composition; `*_sum` = plain sum; `ml_kcl_<node>` = Kirchhoff residual of the plain model currents) | `Results/gate_ml_results.csv` |
| SPICE and ML side by side | `Results/gate_spice_vs_ml.csv` |
| Results tables (deliverable 6) | `Results/gate_results_table_total.csv`, `Results/gate_results_table_vdd.csv` |
| Kirchhoff composition vs plain sum | `Results/gate_composition_comparison.csv` |
| Parity plots (deliverable 7) | `Plots/gate_parity_total.png`, `Plots/gate_parity_vdd.png` |
| Error against temperature and VDD | `Plots/gate_error_vs_conditions.png` |
| Per-transistor diagnostic (12 corners) | `Results/diag_per_transistor.csv`, `Results/diag_error_by_vds.csv` |

**Open point:** section 5 asks to record "the internal node voltages from both" SPICE and ML. Step 4 uses the SPICE node voltages
(as step 4 of the assignment says); `ml_v_<node>` repeats them. An ML-only estimate would solve Kirchhoff for the node voltages
with the models — worth confirming with the TAs whether that is expected.
