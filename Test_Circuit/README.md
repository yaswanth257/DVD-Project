# Part C — Test circuit: 3-bit carry-lookahead adder (CLA)

The leakage of a complete circuit, predicted hierarchically (transistor models → gate models of Part B → circuit) and compared
with a SPICE simulation of the full transistor-level netlist: 128 input vectors × the same 200 PVT corners as Part B =
**25,600 operating points**.

**Result (total leakage): median error 0.25 %, 95th percentile 0.79 %, bias −0.23 %, mean accuracy 99.68 %.**

## Circuit

The adder computes S[2:0] and Cout from A[2:0], B[2:0] and Cin (7 primary inputs a0 – a2, b0 – b2, cin; 128 input vectors).
It is built only from gates validated in Part B (NAND2, AND2, OR2), so that every gate has a gate model:

| Signal | Logic | Gates |
|---|---|---|
| generate `g_i` | `a_i · b_i` | 3 AND2 |
| propagate `p_i` | `a_i ⊕ b_i` | 3 XOR = 12 NAND2 (4-NAND XOR) |
| carry `c1` | `g0 + p0·cin` | 1 AND2, 1 OR2 |
| carry `c2` | `g1 + p1·g0 + p1·p0·cin` | 3 AND2, 2 OR2 |
| carry `cout` | `g2 + p2·g1 + p2·p1·g0 + p2·p1·p0·cin` | 5 AND2, 3 OR2 |
| sum `s_i` | `p_i ⊕ c_i` (`c0 = cin`) | 3 XOR = 12 NAND2 |

Every carry is computed directly from the generate / propagate bits and cin (two-level AND-OR built from 2-input gates), so no
carry waits for the previous one — the carry-lookahead structure. In total **42 gates** (12 AND2, 24 NAND2, 6 OR2), 204
transistors. `circuit_common.check_adder()` verifies on import that the gate list adds correctly for all 128 input vectors.
The instance list (names, inputs, outputs) is `GATES` in `Scripts/circuit_common.py`.

Sizing as in Part B (L = 45 nm): NAND2 Wn / Wp = 180 / 180 nm; AND2 = NAND2 + INV (INV 90 / 180 nm); OR2 = NOR2 (90 / 360 nm) + INV.
NMOS bodies at 0, PMOS bodies at VDD; every netlist has the required `.options` line.

## Method

**SPICE reference (step 8).** One complete transistor-level netlist per corner; the 128 input vectors are applied with
`alter` + `op` in one ngspice run. Leakage definition as in Part B:
*total* = current delivered by the VDD source + by the primary-input sources at logic 1.

**ML prediction (step 9), as the assignment's step 7 describes.** For each input vector the circuit logic is evaluated to get the
logic value at the inputs of every gate; for each gate the Part B gate model gives its *total* leakage at that input combination
and corner (`Gate_Validation/Results/gate_ml_results.csv`); the circuit leakage is the sum over the 42 gates. This assumes, as
the assignment says, that every gate input sits at exactly 0 or VDD and that gates do not load each other.

**Why the sum of gate *total* leakages matches the circuit *total*.** A gate's *total* includes the currents of its inputs at
logic 1. Inside the circuit an input at logic 1 is fed by the driving gate's ON PMOS from VDD (part of the circuit's VDD
current), and an input at logic 0 drains into the driving gate's output (counted in the receiving gate's own VDD current), so
each input current is counted once, in the gate it belongs to. Only *total* leakage is reported for the circuit, for that reason.

## Results (deliverable 8)

| Leakage | SPICE median (A) | ML median (A) | Bias | Median error | 95th pct error | Mean accuracy |
|---|---:|---:|---:|---:|---:|---:|
| total | 7.564 × 10⁻⁶ | 7.546 × 10⁻⁶ | −0.23 % | 0.25 % | 0.79 % | 99.68 % |

Per input vector (200 corners each; `Results/cla_results_table.csv`, the same columns as the gate results table): median error
0.19 – 0.32 %, bias −0.16 to −0.29 %, 95th percentile at most 0.93 %; the largest single error is 1.22 %. Most accurate: `0010010`
(0.19 %), `0000000`, `0110110`; least accurate: `1100011`, `1110001`, `1010101` (0.32 %). The circuit leakage spans
1.1e-6 – 3.6e-5 A over the corners. Parity plot (log axes): `Plots/cla_parity.png`.

**Compared with the gate level** (Part B: median 0.18 %, bias −0.07 %) the circuit is a little less accurate, as the assignment
expects. The error splits into two parts of similar size:

| 25,600 points | Bias | Median error | 95th pct error |
|---|---:|---:|---:|
| sum of the gates' SPICE leakages vs the full-circuit SPICE (composition / loading only, no ML) | −0.11 % | 0.12 % | 0.65 % |
| sum of the gates' ML leakages vs the sum of their SPICE leakages (ML only) | −0.10 % | 0.14 % | 0.50 % |
| sum of the gates' ML leakages vs the full-circuit SPICE (deliverable 8) | −0.23 % | 0.25 % | 0.79 % |

The composition part comes from the assumptions above: inside the circuit a gate input is a gate output a few µV – mV away from
the rail, not an ideal source. Both parts slightly underestimate the leakage, and the error grows with temperature (median
error 0.13 % below 0 °C, 0.53 % above 85 °C), where the leakage itself is largest.

## Scripts

Run from the repository root, in order, after Part B (step 9 reads `Gate_Validation/Results/gate_ml_results.csv`).

| Script | What it does |
|---|---|
| `circuit_common.py` | CLA gate list, logic evaluator, adder self-check, file paths |
| `07_write_cla_netlist.py` | `Netlists/CLA3.cir`: the full transistor-level netlist at corner 0, inputs all 0 (deliverable 2) |
| `08_run_spice_cla.py [--workers 8]` | ngspice, 128 input vectors × 200 corners → `Results/cla_spice_results.csv` (about 20 minutes with 8 workers) |
| `09_predict_ml_cla.py` | ML prediction by summing the Part B gate leakages → `Results/cla_ml_results.csv` |
| `10_analyze_cla.py` | `Results/cla_results_table.csv` (per input vector + overall) and `Plots/cla_parity.png` |

## Files

```
Test_Circuit/
  Scripts/    circuit_common.py, 07_write_cla_netlist.py, 08_run_spice_cla.py, 09_predict_ml_cla.py, 10_analyze_cla.py
  Netlists/   CLA3.cir                 full transistor-level netlist (corner 0, all inputs 0)
  Results/    cla_spice_results.csv    SPICE leakage, 128 input vectors × 200 corners
              cla_ml_results.csv       ML leakage
              cla_results_table.csv    per-input-vector statistics + overall row (deliverable 8)
  Plots/      cla_parity.png           ML vs SPICE, log axes (deliverable 8)
```

*History:* the first version of this circuit computed the carries one after another (`c_{i+1} = g_i + p_i·c_i`, 33 gates), which
is a ripple-carry chain in generate / propagate form; it was replaced by the lookahead carries above. Its accuracy was the same
(median 0.25 %, mean accuracy 99.68 %).
