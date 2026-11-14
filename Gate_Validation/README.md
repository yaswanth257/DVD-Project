# Part B — Gate-level validation (INV, NAND2, NOR2, AND2, OR2)

Gate leakage predicted from the four transistor models, compared with transistor-level ngspice simulations of the gates.
18 gate / input cases (INV: 2, NAND2, NOR2, AND2, OR2: 4 each) × 200 PVT corners = **3,600 operating points**.

**Result (total leakage, all 3,600 points): median error 0.18 %, 95th percentile 0.75 %, bias −0.07 %, mean accuracy 99.74 %.**

## Workflow

Run from the repository root, in order (seed `12175305`; `ngspice` must be on the path).

| Step | Script | Output |
|---|---|---|
| 1 | `01_generate_corners.py` | `Results/corners.csv` — 200 PVT corners |
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

All four transistor models were also trained with 4,000 extra single-transistor rows per case from the region where the transistor
operates inside gates (`Data_Generation/Scripts/generate_gate_region_data.py`, own random draws, no gate result used). ON models:
both terminals near a rail, |Vds| down to 0.1 µV. OFF models: drain and source at opposite rails, at the same rail, or one at a rail
and one at a series-stack node (the cells that OFF transistors occupy in these gates). Uniform sampling of vd / vs (section 4) almost never produces such points, and there the ON models' `ig`
and `ib` — the currents the Kirchhoff composition does use — were 2 – 7 % off; now 0.2 – 0.4 %
(`ML_Models/README.md`, "Final models"). The gate SPICE results of step 3 are used for evaluation only.

## Results (deliverable 6)

| Leakage | SPICE median (A) | ML median (A) | Bias | Median error | 95th pct error | Mean accuracy |
|---|---:|---:|---:|---:|---:|---:|
| total | 1.552 × 10⁻7 | 1.550 × 10⁻7 | −0.07 % | 0.18 % | 0.75 % | 99.74 % |
| vdd | 1.201 × 10⁻7 | 1.198 × 10⁻7 | −0.10 % | 0.22 % | 1.02 % | 99.66 % |

Error = ML / SPICE − 1; bias = median signed error; mean accuracy = 100 % − mean |error|. Per gate and input, total leakage
(200 corners each; `Results/gate_results_table_total.csv`, the *vdd* table in `gate_results_table_vdd.csv`):

| Gate | Input | SPICE median (A) | ML median (A) | Bias | Median error | 95th pct error | Mean accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| INV | 0 | 6.669e-08 | 6.660e-08 | +0.04 % | 0.20 % | 0.79 % | 99.72 % |
| INV | 1 | 8.163e-08 | 8.145e-08 | −0.05 % | 0.16 % | 0.61 % | 99.79 % |
| NAND2 | 00 | 9.277e-08 | 9.292e-08 | +0.01 % | 0.22 % | 0.70 % | 99.74 % |
| NAND2 | 01 | 1.356e-07 | 1.354e-07 | −0.09 % | 0.17 % | 0.64 % | 99.77 % |
| NAND2 | 10 | 7.543e-08 | 7.534e-08 | −0.04 % | 0.18 % | 0.64 % | 99.74 % |
| NAND2 | 11 | 2.136e-07 | 2.134e-07 | −0.02 % | 0.15 % | 0.51 % | 99.80 % |
| NOR2 | 00 | 2.100e-07 | 2.098e-07 | −0.16 % | 0.32 % | 0.99 % | 99.62 % |
| NOR2 | 01 | 2.443e-07 | 2.437e-07 | −0.23 % | 0.26 % | 0.86 % | 99.66 % |
| NOR2 | 10 | 1.141e-07 | 1.139e-07 | −0.05 % | 0.17 % | 0.88 % | 99.71 % |
| NOR2 | 11 | 7.791e-08 | 7.833e-08 | +0.03 % | 0.22 % | 1.32 % | 99.63 % |
| AND2 | 00 | 1.977e-07 | 1.983e-07 | −0.05 % | 0.14 % | 0.55 % | 99.81 % |
| AND2 | 01 | 2.197e-07 | 2.197e-07 | −0.15 % | 0.20 % | 0.56 % | 99.77 % |
| AND2 | 10 | 1.548e-07 | 1.548e-07 | −0.13 % | 0.18 % | 0.63 % | 99.77 % |
| AND2 | 11 | 2.791e-07 | 2.782e-07 | −0.01 % | 0.12 % | 0.47 % | 99.83 % |
| OR2 | 00 | 3.135e-07 | 3.120e-07 | −0.25 % | 0.28 % | 0.81 % | 99.67 % |
| OR2 | 01 | 3.124e-07 | 3.117e-07 | −0.20 % | 0.22 % | 0.73 % | 99.72 % |
| OR2 | 10 | 1.775e-07 | 1.775e-07 | −0.04 % | 0.14 % | 0.74 % | 99.79 % |
| OR2 | 11 | 1.452e-07 | 1.456e-07 | +0.01 % | 0.14 % | 0.84 % | 99.77 % |

Most accurate: AND2 `11` (0.12 %), AND2 `00` and OR2 `11` (0.14 %); least accurate: NOR2 `00` (median 0.32 %, 95th percentile
0.99 %) and OR2 `00` (0.28 %). On the *vdd* definition NOR2 `11` is the weakest case (median 0.72 %, 95th percentile 4.1 %): its
VDD current is only 1.3e-8 A, the leakage through two series OFF PMOS (about 10× below the median case), so the same absolute
model error is a larger relative one. All other cases are ≤ 0.32 % (median).

## How the error was brought down

| Total leakage, 3,600 points | Bias | Median error | 95th pct | Mean accuracy |
|---|---:|---:|---:|---:|
| Plain sum of the model currents, earlier ON models (first version of Part B) | −3.31 % | 4.22 % | 13.87 % | 94.59 % |
| + Kirchhoff for the ON channels | +1.33 % | 1.42 % | 7.29 % | 97.71 % |
| + ON models trained with gate-region rows | +0.03 % | 0.34 % | 1.64 % | 99.47 % |
| + OFF models trained with gate-region rows (final) | **−0.07 %** | **0.18 %** | **0.75 %** | **99.74 %** |
| (plain sum with the final models, for comparison) | +2.78 % | 6.35 % | 36.27 % | 89.44 % |

- **Step 6 shows where the error came from.** Summing standalone single-transistor SPICE currents, at the voltages each transistor
  sees inside the gate, reproduces the gate SPICE leakage to a median of 0.035 % (the composition and the signs are right). The
  model errors inside gates are now small for |Vds| > 0.1 V (OFF transistors at |Vds| > 0.5 V: median 0.29 % NMOS / 0.28 % PMOS,
  95th percentile 1.7 % / 1.1 %; before the OFF gate-region rows 0.41 % / 0.67 % and 3.3 % / 2.1 %) and large only for ON
  transistors at |Vds| < 1 mV (Kirchhoff takes their channel current) and for the untrained `id` / `is` of OFF transistors at
  |Vds| < 10 mV (above).
- **The first version's errors followed one pattern:** they were large exactly when an ON transistor sits on the VDD net (INV `0`,
  NAND2 `00`, NOR2 `00` / `01`: 8 – 12 %) and small otherwise (≤ 1.8 %), because the supply current then flows through that ON
  channel.
- **After Kirchhoff, the ON transistors' `ig` / `ib` were the remaining source:** replacing only those with single-transistor SPICE
  values (12 corners) brought the median from 1.74 % to 0.27 %. The gate-region rows fixed exactly that (median `ig` / `ib` error at
  |Vds| < 1 mV: NMOS 1.8 / 2.4 % → 0.18 / 0.36 %, PMOS 4.4 / 6.6 % → 0.34 / 0.39 %, final models).
- **Then the OFF channel currents were the remaining source.** On the 12 diagnostic corners (216 points) with the ON-fixed models, the
  true (single-transistor SPICE) values were swapped in for one group of model currents at a time and the gate leakage recomposed:

  | Replaced by SPICE values | median error | 95th pct | mean |
  |---|---:|---:|---:|
  | nothing (models after the ON fix) | 0.41 % | 1.34 % | 0.53 % |
  | `ig` / `ib` of the ON transistors | 0.27 % | 1.38 % | 0.46 % |
  | all currents of OFF transistors at \|Vds\| > 0.1 V | 0.12 % | 0.62 % | 0.19 % |
  | all currents of OFF transistors at \|Vds\| < 0.1 V | 0.38 % | 1.34 % | 0.50 % |
  | `ig` / `ib` of the OFF transistors only | 0.39 % | 1.36 % | 0.52 % |

  The OFF transistors' channel current at high |Vds| is what the gate leaks. Gates place OFF transistors at the corner of the (vd, vs)
  square (52 % of the 8,400 OFF instances: drain and source at opposite rails, 19 % at the same rail, 29 % one at a rail and one at a
  stack node), where 20,000 uniform rows have about 8 rows within 2 % of VDD of the opposite-rail corner. 4,000 rows per OFF case from
  these cells (`generate_gate_region_data.py --cases OFF`) halved the gate error again: median 0.34 % → 0.18 %, 95th percentile
  1.64 % → 0.75 %. The `id` / `is` of the OFF models at |Vds| < 10 mV are, like the ON models', not in the loss and are now much worse
  than before (median error 38 – 65 % on the 164 such instances of the 12 diagnostic corners), which the swap table shows does not
  matter for the leakage; that is also why the plain sum of the final models is poor (table above).

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
