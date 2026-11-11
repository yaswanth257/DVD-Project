# Part B — Gate-level validation (INV, NAND2, NOR2, AND2, OR2)

Part B compares gate leakage estimated using the trained transistor models against transistor-level ngspice simulations.

Run the commands from the repository root, in order. All scripts use the fixed seed `12175305`. The simulations require `ngspice`.

## Workflow

| Step | Script | Output |
|---|---|---|
| 1 | `01_generate_corners.py` | `Results/corners.csv` — 200 PVT corners |
| 2 | `02_write_gate_netlists.py` | `Netlists/{INV,NAND2,NOR2,AND2,OR2}.cir` — transistor-level gate netlists |
| 3 | `03_run_spice_gates.py` | `Results/gate_spice_results.csv` — SPICE leakage, internal-node and output voltages |
| 4 | `04_predict_ml_gates.py` | `Results/gate_ml_results.csv` — ML leakage predictions using the four trained transistor models |
| 5 | `05_analyze_gates.py` | Comparison tables in `Results/` and parity/error plots in `Plots/` |
| 6 | `06_diagnose_models_in_gates.py 12` | Diagnostic results in `Results/` using 12 corners |

A total of 18 gate/input combinations were evaluated: 2 for INV and 4 each for NAND2, NOR2, AND2 and OR2. With 200 corners per combination, this gives **3,600 operating points**.

## Setup choices

- **Technology:** L = 45 nm; transistor widths are within the trained range of 45–360 nm.
- **Sizing:**
  - INV: Wn/Wp = 90/180 nm
  - NAND2: 180/180 nm
  - NOR2: 90/360 nm
  - AND2 and OR2 use the corresponding NAND2/NOR2 stage followed by an inverter.
- **Topology:** NAND2 uses a series NMOS network and parallel PMOS network. NOR2 uses a parallel NMOS network and series PMOS network.
- **Complete gates:** AND2 and OR2 are simulated as complete transistor-level gate netlists, including their inverter stages.
- **Leakage definitions:** Both total leakage and VDD-only leakage are reported.
  - **Total leakage:** current supplied by VDD plus current supplied by input sources held at logic 1.
  - **VDD leakage:** current supplied by the VDD source only.
- **ML prediction:** The transistor model is selected according to the transistor type and gate voltage. Terminal voltages are taken from the SPICE operating points and clipped to the range [0, VDD].

## Results

The following metrics cover all 3,600 operating points.

| Leakage definition | SPICE median (A) | ML median (A) | Bias | Median error | 95th percentile error | Mean accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Total | 1.552 × 10⁻⁷ | 1.477 × 10⁻⁷ | -3.31% | 4.22% | 13.87% | 94.59% |
| VDD only | 1.201 × 10⁻⁷ | 1.127 × 10⁻⁷ | -4.12% | 4.97% | 14.85% | 94.00% |

For total leakage, some of the lowest median errors were:

- NOR2 input `10`: 0.63%
- INV input `1`: 0.86%
- NAND2 input `11`: 1.07%

Some of the larger median errors were:

- NOR2 input `00`: 11.85%
- NOR2 input `01`: 10.14%

The per-gate and per-input-combination metrics are available in the result tables.

## Diagnostic analysis

The diagnostic script was run using 12 corners.

Summing the individual transistor SPICE currents at the gate operating-point voltages reproduced the gate-level SPICE leakage with a median absolute difference of **0.0354%** and a maximum difference of **0.713%** over the checked cases.

The transistor-level comparison showed larger relative errors for some transistor states at very small `|VDS|`. Errors were generally smaller at higher `|VDS|`.

## Generated files

| Description | File |
|---|---|
| Generated PVT corners | `Results/corners.csv` |
| SPICE gate results | `Results/gate_spice_results.csv` |
| ML gate predictions | `Results/gate_ml_results.csv` |
| Combined SPICE and ML results | `Results/gate_spice_vs_ml.csv` |
| Total leakage result table | `Results/gate_results_table_total.csv` |
| VDD leakage result table | `Results/gate_results_table_vdd.csv` |
| Total leakage parity plot | `Plots/gate_parity_total.png` |
| VDD leakage parity plot | `Plots/gate_parity_vdd.png` |
| Error versus PVT conditions | `Plots/gate_error_vs_conditions.png` |
| Per-transistor diagnostic results | `Results/diag_per_transistor.csv` |
| Error grouped by transistor state and VDS | `Results/diag_error_by_vds.csv` |
