# Data generation

Single-transistor datasets for the four cases NMOS ON, NMOS OFF, PMOS ON and PMOS OFF.
Every sample is one ngspice DC operating point of one transistor with its own voltage source on each terminal.

| Setting | Value |
|---|---|
| Model file | `45nm_MGK.pm` (PTM 45 nm metal gate / high-k, BSIM4 level 54) |
| Nominal VDD | 0.9 V (drawn uniformly in 0.81 – 0.99 V) |
| Channel length / width | L = 45 nm, W uniform in 45 – 360 nm (1 – 8 × Lmin) |
| Temperature | uniform in −40 … 125 °C |
| Process parameters | `vth0`, `toxe`, `ndep`, `u0`, `xj`: uniform in nominal ± 10 % (separate NMOS / PMOS nominals read from the model file) |
| Terminal voltages | `vd`, `vs` uniform in 0 … VDD; `vg`, `vb` fixed by the case (below) |
| Random seed | **12175305** (`np.random.default_rng`), used by every script |
| Samples | **20,000 per case** |
| Simulator options | `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |

| Case | `vg` | `vb` |
|---|---|---|
| NMOS ON | VDD | 0 |
| NMOS OFF | 0 | 0 |
| PMOS ON | 0 | VDD |
| PMOS OFF | VDD | VDD |

Nominal values read from `45nm_MGK.pm`:

| | `vth0` (V) | `toxe` (m) | `ndep` (cm⁻³) | `u0` (m²/V·s) | `xj` (m) |
|---|---|---|---|---|---|
| NMOS | 0.3423 | 9.0e-10 | 6.5e18 | 0.02947 | 1.4e-8 |
| PMOS | −0.23122 | 9.2e-10 | 2.8e18 | 0.00391 | 1.4e-8 |

## Files

```
Datasets/
  NMOS_ON.csv  NMOS_OFF.csv  PMOS_ON.csv  PMOS_OFF.csv            original 10,000-sample datasets
  NMOS_ON_20k.csv  NMOS_OFF_20k.csv  PMOS_ON_20k.csv  PMOS_OFF_20k.csv   20,000-sample datasets used for training
  NMOS_ON_gate_region.csv  PMOS_ON_gate_region.csv                  4,000 extra rows each for the ON models (gate operating region)
  NMOS_OFF_gate_region.csv  PMOS_OFF_gate_region.csv                4,000 extra rows each for the OFF models (gate operating region)
Scripts/        generation and simulation scripts (table below)
Netlists/       NMOS_ON.cir  NMOS_OFF.cir  PMOS_ON.cir  PMOS_OFF.cir   example single-transistor netlists (deliverable 2)
Data_Check/     data_check.ipynb and its outputs (Kirchhoff check, trend plots, dataset summary)
```

Columns of every CSV, in this order: `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib`
(voltages in V, `temp` in °C, `w` / `toxe` / `xj` in m, currents in A; the currents are the ngspice currents of the four
terminal voltage sources, so their sign gives the direction).

## How the 20,000 samples are built

The 10k scripts draw, from a fresh `default_rng(12175305)`, ten arrays of 10,000 values in this order:
`vdd → vd → vs → temp → w → vth0 → toxe → ndep → u0 → xj`. The 20k datasets **extend** that sequence instead of drawing a new
one:

| Rows | Content |
|---|---|
| 1 – 10,000 | identical to the original 10k CSV (same seed, same draws, same SPICE results) |
| 10,001 – 20,000 | a second block of 10,000 draws that continues the same random stream, so no sample repeats |

Each case uses its own fresh generator, so the ON and OFF datasets of one device type share the same 20,000 input points
(only `vg` differs). NMOS and PMOS use the same construction (`generate_nmos_20k.py`, `generate_pmos_20k.py`).
`generate_nmos_20k.py` additionally **fails** if rows 1 – 10,000 do not reproduce the original 10k CSV (inputs exactly,
currents to 6 digits); the run that produced the committed files reproduced them exactly (relative difference 0.0).

## Scripts

| Script | Purpose |
|---|---|
| `generate_nmos_on_inputs.py`, `generate_nmos_off_inputs.py` | draw the 10k NMOS inputs |
| `simulate_nmos_on.py`, `simulate_nmos_off.py` | simulate them → `NMOS_ON.csv`, `NMOS_OFF.csv` |
| `generate_pmos_datasets.py` | draw + simulate the 10k PMOS cases → `PMOS_ON.csv`, `PMOS_OFF.csv` |
| `generate_nmos_20k.py` | NMOS 20k datasets (ON and OFF) |
| `generate_pmos_20k.py` | PMOS 20k datasets (ON and OFF) |
| `write_example_netlists.py` | writes `Netlists/<case>.cir` (the netlist of row 1 of each dataset) and checks it in ngspice against the CSV |
| `generate_gate_region_data.py` | `{NMOS,PMOS}_{ON,OFF}_gate_region.csv`: extra training rows where the transistors operate inside gates (below); `--cases ON|OFF|all` |

Run everything from the repository root (the netlists include `45nm_MGK.pm` by relative path):

```bash
python3 Data_Generation/Scripts/generate_nmos_20k.py      # 2 × 20,000 simulations, about 20 min here (8 cores, one simulation at a time)
python3 Data_Generation/Scripts/generate_pmos_20k.py
python3 Data_Generation/Scripts/write_example_netlists.py # the four example netlists (seconds)
python3 Data_Generation/Scripts/generate_gate_region_data.py   # 4 × 4,000 simulations (--cases ON or OFF for two of them); ON about 15 s with 6 processes, OFF about 5 min
```

## Gate-region rows (`*_ON_gate_region.csv`, `*_OFF_gate_region.csv`)

Inside a static CMOS gate an ON transistor connects a node to a rail: both terminals sit near the same rail and |Vds| is only
20 µV – 1 mV. Uniformly drawn vd and vs (section 4) almost never produce such points (|Vds| < 1 mV in 0.2 % of the rows), and
there the ON models' gate and body currents were 2 – 7 % off, which set most of the gate-level error of Part B. These files add
4,000 rows per ON case with VDD, temperature, W and the process parameters drawn as in section 4 and only vd / vs drawn
differently: `vs` is near 0, near VDD or uniform (1/3 each; distance to the rail log-uniform in 0.1 µV – 0.45 V); `vd` is either
on the diagonal (`vd = vs ± 10^U(−7, 0)` V) or drawn like `vs` (1/2 each). Same single-transistor netlist, own random draws
(seed 12175305, a fresh generator per case), no gate simulation result. About one third of the rows have |Vds| < 1 mV. The
training script applies the data-check rule (no zero current, Kirchhoff residual ≤ 1e-3 of the largest current); it keeps 3,911
NMOS_ON and 3,996 PMOS_ON rows — the others have all terminals within microvolts of each other, with currents at the solver's
noise floor.

OFF cases. Once the ON models were fixed, the gate-level error that remained sat in the channel current of OFF transistors at
|Vds| > 0.1 V (see `Gate_Validation/README.md`). Gates put OFF transistors at the corner of the (vd, vs) square: of the 8,400
OFF-transistor instances of the 18 gate cases × 200 corners, 52 % have drain and source at opposite rails, 19 % both at the same
rail and 29 % one at a rail and one at a series-stack node, while the 20,000 uniform rows contain about 8 rows within 2 % of VDD of
the opposite-rail corner. The OFF files add 4,000 rows per case, again with everything as in section 4 except vd / vs; for
NMOS_OFF (PMOS_OFF mirrored, vd → VDD − vd, vs → VDD − vs): 45 % opposite rails (`vd = VDD − o`, `vs = o'`), 20 % same rail
(`vs = o`, `vd = vs ± 10^U(−7, 0)` V), 20 % `vs = o` with `vd` uniform, 15 % `vd = VDD − o` with `vs` uniform; the distance `o`
to a rail is log-uniform in 0.1 µV – 0.45 V. Own random draws (seed 12175305, a fresh generator per case), no gate simulation
result. All 4,000 NMOS_OFF rows and 3,915 PMOS_OFF rows pass the data-check rule above. These rows are not drawn from the section 4
distribution and were chosen after looking at the gate-level evaluation; the report states this. The 10k and 20k datasets
(deliverable 1) are unchanged.

## Example netlists (`Netlists/`)

One single-transistor netlist per case: exactly the netlist the generators simulate for every sample, filled in with row 1
of the dataset (its SPICE currents are in the header comment). The model is included relative to the netlist file, so each
one runs from any directory: `ngspice -b Data_Generation/Netlists/NMOS_OFF.cir` prints `RESULT id ig is ib`. Re-simulated with
ngspice 42, all four reproduce the currents stored in the CSVs exactly.

## Data check (`Data_Check/data_check.ipynb`)

All checks pass for all four 20k datasets:

* **Dataset summary:** 20,000 rows, columns in spec order, no missing / infinite values, no duplicate rows, no zero currents,
  every input within its specified range, `vg` / `vb` as required by the case, and rows 1 – 10,000 equal to the original 10k dataset.
* **Kirchhoff check** (`id + ig + is + ib ≈ 0`, relative to the largest current of the sample; limit 1e-3):

| Case | max abs residual (A) | median relative | max relative |
|---|---|---|---|
| NMOS_ON | 9.9e-10 | 8.9e-7 | 1.5e-4 |
| NMOS_OFF | 8.4e-13 | 8.2e-7 | 8.5e-6 |
| PMOS_ON | 9.7e-10 | 8.1e-7 | 9.1e-6 |
| PMOS_OFF | 9.1e-13 | 9.8e-7 | 2.2e-4 |

* **Trend plots** (`fig2` – `fig4`): OFF current rises with temperature, falls with |vth0|, and the gate current falls with
  `toxe` — 12 / 12 trend checks agree with the expected physics for all four cases.

No rows are removed or filtered: every row is a valid SPICE result. Rows where one terminal current is almost cancelled by the
others are real physics; they matter for the model error (see `ML_Models/README.md`).

## Notes

1. **NMOS 20k files were regenerated.** The previous `NMOS_*_20k.csv` were built by running the 10k generator with
   `NUM_SAMPLES = 20000` (as the old data-check notebook noted), i.e. 20,000 values per variable in one block; the committed
   scripts still say 10,000, so those files could not be reproduced from the repo. They were valid random samples but shared
   **no row** with the 10k datasets (only the `vdd` column coincided), unlike the PMOS 20k files. They were replaced by files
   built exactly like the PMOS ones (table above). The old files remain in the git history.
2. **Removed PMOS per-case scripts.** `generate_pmos_on_inputs.py`, `generate_pmos_off_inputs.py`, `simulate_pmos_on.py` and
   `simulate_pmos_off.py` were removed: they drew the negative `vth0` as `−U(0.9|v|, 1.1|v|)` instead of `U(1.1v, 0.9v)`, so they
   did not reproduce the committed `PMOS_*.csv` (same range, mirrored draw). `generate_pmos_datasets.py` (10k) and
   `generate_pmos_20k.py` reproduce them exactly. The old scripts remain in the git history.
3. **ngspice executable name.** Every script uses `ngspice_con` (the Windows console build) if it exists and plain `ngspice`
   otherwise (`generate_pmos_datasets.py` always calls `ngspice`), so they run unchanged on Windows and Linux. The NMOS 20k files were generated with ngspice 45.2, which
   reproduced the stored 10k currents exactly.
