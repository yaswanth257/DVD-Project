# Test Circuit - Part C: 3-bit Carry-Lookahead Adder (CLA)

Part C applies the ML transistor models from Part A and the gate-level leakage estimates
from Part B to predict the leakage current of a complete combinational circuit across all
input combinations and all 200 PVT corners, and validates the prediction against SPICE.

***

## Circuit description

The 3-bit CLA adds two 3-bit operands A[2:0] and B[2:0] with carry-in Cin and produces
sum S[2:0] and carry-out Cout.

| Primary inputs | a0, a1, a2, b0, b1, b2, cin (7 inputs, 128 combinations) |
| - | - |
| Primary outputs | s0, s1, s2, cout_net |
| Total gates | 6 AND2 + 24 NAND2 + 3 OR2 = 33 gates |

XOR is built from four NAND2 gates (classic 4-NAND XOR). Gate sizing follows
gate_common.py (45 nm process, L = 45 nm):

| Gate | Wn | Wp |
| - | - | - |
| NAND2 | 180 nm | 180 nm |
| AND2 (NAND2 + INV) | 180/90 nm | 180/180 nm |
| OR2 (NOR2 + INV) | 90/90 nm | 360/180 nm |

Gate instance list (circuit_common.py GATES):

| Instance | Type | Inputs | Output |
| - | - | - | - |
| and_g0 | AND2 | a0, b0 | g0 |
| and_g1 | AND2 | a1, b1 | g1 |
| and_g2 | AND2 | a2, b2 | g2 |
| xp0_n1..xp0_n4 | NAND2 x4 | a0, b0 (XOR) | p0 |
| xp1_n1..xp1_n4 | NAND2 x4 | a1, b1 (XOR) | p1 |
| xp2_n1..xp2_n4 | NAND2 x4 | a2, b2 (XOR) | p2 |
| and_pc0 | AND2 | p0, cin | pc0 |
| or_c1 | OR2 | g0, pc0 | c1 |
| and_pc1 | AND2 | p1, c1 | pc1 |
| or_c2 | OR2 | g1, pc1 | c2 |
| and_pc2 | AND2 | p2, c2 | pc2 |
| or_cout | OR2 | g2, pc2 | cout_net |
| xs0_n1..xs0_n4 | NAND2 x4 | p0, cin (XOR) | s0 |
| xs1_n1..xs1_n4 | NAND2 x4 | p1, c1 (XOR) | s1 |
| xs2_n1..xs2_n4 | NAND2 x4 | p2, c2 (XOR) | s2 |

***

## ML prediction method

The CLA ML leakage for one input combination at one corner is the sum of the
ml_leak_total values of all 33 gate instances, looked up from the Part B
gate_ml_results.csv by (gate_type, gate_input_combo, corner). This is valid under
the assignment assumption that every gate input sits at exactly 0 V or VDD. No new
SPICE runs are needed, and script 09 evaluates the combinational logic to find which
(gate_type, input_combo) each gate instance uses at every CLA input combination.

***

## Scripts

| Script | What it does |
| - | - |
| circuit_common.py | CLA gate list, logic evaluator, file paths |
| 07_write_cla_netlist.py | Writes the full transistor-level netlist (deliverable 2) |
| 08_run_spice_cla.py | ngspice: 128 combos x 200 corners (25,600 op points) |
| 09_predict_ml_cla.py | ML prediction by summing gate leakages from Part B |
| 10_analyze_cla.py | Merge, error stats, results table, parity plot |

Run from the repository root in order 07 to 10.

***

## Results

Seed 12175305, 200 PVT corners, 128 input combinations, 25,600 total operating points.

| Leakage | Bias | Median error | 95th pct error | Mean accuracy |
| - | - | - | - | - |
| total (VDD source + inputs at logic 1) | -0.229 % | 0.251 % | 0.802 % | 99.68 % |

Full per-input-combo table: [Results/cla_results_table.csv](Results/cla_results_table.csv)

Parity plot (log axes): [Plots/cla_parity.png](Plots/cla_parity.png)

***

## Files

```
Test_Circuit/
  Scripts/
    circuit_common.py        CLA topology and logic evaluator
    07_write_cla_netlist.py  example netlist writer (deliverable 2)
    08_run_spice_cla.py      SPICE reference simulation
    09_predict_ml_cla.py     ML leakage prediction
    10_analyze_cla.py        analysis and plots
  Netlists/
    CLA3.cir                 transistor-level example netlist (corner 0, all inputs 0)
  Results/
    cla_spice_results.csv    SPICE leakage (128 combos x 200 corners)
    cla_ml_results.csv       ML leakage predictions
    cla_results_table.csv    per-combo error statistics + overall row
  Plots/
    cla_parity.png           predicted vs SPICE parity plot (log axes)
```
