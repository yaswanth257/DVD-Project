# PMOS Dataset Generation — README

**Author:** Sangameshwar  
**Date:** 2026-11-05  
**Project:** DVD (Design Verification & Design) — Leakage Current Estimation using ML  
**Random Seed:** `12175305`

---

## Overview

This document describes the generation of two PMOS simulation datasets:

| Dataset | File | Samples |
|---------|------|---------|
| PMOS ON  | `Data_Generation/Datasets/PMOS_ON.csv`  | 10,000 |
| PMOS OFF | `Data_Generation/Datasets/PMOS_OFF.csv` | 10,000 |

These datasets mirror the existing NMOS ON/OFF datasets in structure and methodology, adapted for a **PTM 45nm Metal Gate / High-K PMOS transistor** using the model file `45nm_MGK.pm`.

---

## Directory Structure

```
DVD-Project/
├── 45nm_MGK.pm                          # BSIM4 model file (NMOS + PMOS)
├── README_sangameshwar_pmos_dataset.md  # This file
└── Data_Generation/
    ├── Scripts/
    │   ├── generate_pmos_on_inputs.py   # Input generator — PMOS ON  (pandas-based)
    │   ├── generate_pmos_off_inputs.py  # Input generator — PMOS OFF (pandas-based)
    │   ├── simulate_pmos_on.py          # ngspice runner  — PMOS ON
    │   ├── simulate_pmos_off.py         # ngspice runner  — PMOS OFF
    │   └── generate_pmos_datasets.py    # Standalone runner (numpy + csv only)
    └── Datasets/
        ├── PMOS_ON.csv                  # 10,000-row final dataset
        └── PMOS_OFF.csv                 # 10,000-row final dataset
```

---

## Transistor Model — PMOS Nominal Parameters

Extracted from the **PMOS section** of `45nm_MGK.pm` (PTM 45nm node, BSIM4 Level 54):

| Parameter | Symbol | Nominal Value | Description |
|-----------|--------|--------------|-------------|
| Threshold voltage | `vth0` | **−0.23122 V** | Negative for PMOS |
| Gate oxide thickness (electrical) | `toxe` | 9.2 × 10⁻¹⁰ m | Slightly thicker than NMOS |
| Channel doping | `ndep` | 2.8 × 10¹⁸ cm⁻³ | Lower than NMOS |
| Low-field mobility | `u0` | 0.00391 m²/V·s | ~7.5× lower than NMOS (holes) |
| Junction depth | `xj` | 1.4 × 10⁻⁸ m | Same as NMOS |

Each parameter is varied ±10% around nominal using uniform random sampling (process variation).

---

## Input Feature Space

### Fixed Ranges (same for ON and OFF)

| Feature | Min | Max | Description |
|---------|-----|-----|-------------|
| `vdd` | 0.81 V (−10%) | 0.99 V (+10%) | Supply voltage variation |
| `temp` | −40 °C | 125 °C | Operating temperature range |
| `w` | 45 nm (1× L_min) | 360 nm (8× L_min) | Transistor width |
| `vd` | 0 V | `vdd` | Drain voltage |
| `vs` | 0 V | `vdd` | Source voltage |
| `vth0` | 0.9 × nominal | 1.1 × nominal | Process variation |
| `toxe` | 0.9 × nominal | 1.1 × nominal | Process variation |
| `ndep` | 0.9 × nominal | 1.1 × nominal | Process variation |
| `u0` | 0.9 × nominal | 1.1 × nominal | Process variation |
| `xj` | 0.9 × nominal | 1.1 × nominal | Process variation |

> **Note:** Gate length is fixed at `L = 45 nm` (minimum feature size) for all simulations.

### Bias Conditions (key difference: PMOS vs NMOS)

PMOS operates with inverted gate polarity relative to NMOS:

| State | `vg` | `vb` | Physical Reason |
|-------|------|------|----------------|
| **ON**  | `0 V` (VSS) | `vdd` | Vgs = −vdd < Vth → channel inverted, PMOS conducts |
| **OFF** | `vdd`       | `vdd` | Vgs = 0 V > Vth → no inversion, PMOS in cut-off (leakage only) |

The body (`vb`) is always tied to `vdd` (VDD rail) for PMOS, as opposed to NMOS where `vb = 0 V` (GND).

---

## Output Columns (Simulated by ngspice)

| Column | Description |
|--------|-------------|
| `id` | Drain current (A) |
| `ig` | Gate current (A) — includes gate leakage |
| `is` | Source current (A) |
| `ib` | Body current (A) |

Currents are signed (follow convention of ngspice voltage source current: current flowing into the positive terminal is positive).

---

## Full CSV Column Order (16 columns)

```
vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib
```

---

## How It Was Done

### Step 1 — Understand the NMOS Pattern

The existing NMOS scripts (`generate_nmos_on_inputs.py`, `simulate_nmos_on.py`, etc.) were studied carefully. The PMOS scripts follow the identical structure, with adjustments only for:
- PMOS section extraction from the model file
- Bias conditions (`vg`, `vb`) flipped for P-type operation
- Sign handling for the negative `vth0` parameter

### Step 2 — Extract PMOS Parameters from Model File

A regex is used to isolate the PMOS section of `45nm_MGK.pm`:

```python
match = re.search(r"(?ims)^\s*\.model\s+pmos\b(.*)", text)
```

The five process parameters (`vth0`, `toxe`, `ndep`, `u0`, `xj`) are then parsed from that section. Since `vth0` is negative for PMOS, the ±10% variation is computed on `abs(value)` and the sign is restored:

```python
lo = min(0.9 * val, 1.1 * val)   # works correctly for negative values too
hi = max(0.9 * val, 1.1 * val)
process[name] = rng.uniform(lo, hi, NUM_SAMPLES)
```

### Step 3 — Generate 10,000 Random Input Samples

Using `numpy.random.default_rng(seed=12175305)`:

```python
rng  = np.random.default_rng(12175305)
vdd  = rng.uniform(0.81, 0.99, 10000)       # VDD ±10%
vd   = rng.uniform(0, vdd)                  # element-wise upper bound
vs   = rng.uniform(0, vdd)
temp = rng.uniform(-40, 125, 10000)
w    = rng.uniform(45e-9, 360e-9, 10000)
# ... process parameters
```

The same RNG call order is maintained for both ON and OFF (same seed → same input distribution, only `vg`/`vb` differ).

### Step 4 — Set Bias Conditions

```python
# PMOS ON
vg = np.zeros(NUM_SAMPLES)   # gate = 0 V (VSS)
vb = vdd.copy()              # body = VDD

# PMOS OFF
vg = vdd.copy()              # gate = VDD
vb = vdd.copy()              # body = VDD
```

### Step 5 — ngspice Simulation (per sample)

For each of the 10,000 samples, a SPICE netlist is generated on-the-fly:

```spice
* PMOS single-point OP
.include "45nm_MGK.pm"
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.temp <temp>
M1 drain gate source body pmos W=<w> L=45e-9
Vd drain 0 DC <vd>
Vg gate   0 DC <vg>
Vs source 0 DC <vs>
Vb body   0 DC <vb>
.control
  set numdgt=15
  altermod pmos vth0 = <vth0>
  altermod pmos toxe = <toxe>
  ...
  op
  echo RESULT $&id_val $&ig_val $&is_val $&ib_val
.endc
.end
```

`altermod` overrides the BSIM4 parameters per sample without reloading the model, enabling efficient per-sample process variation.

ngspice is invoked in batch mode:
```bash
ngspice -b <netlist.cir>
```

The `RESULT` line is parsed from stdout/stderr with a regex to extract all four terminal currents.

### Step 6 — KCL Validation

After each simulation, Kirchhoff's Current Law is checked:

```
|id + ig + is + ib| → should be ~0
```

| Dataset | Max KCL Residual | Assessment |
|---------|-----------------|------------|
| PMOS_ON  | **9.69 × 10⁻¹⁰ A** (~1 nA) | ✅ Excellent (on-state currents are µA–mA scale) |
| PMOS_OFF | **9.10 × 10⁻¹³ A** (~1 pA) | ✅ Excellent (off-state leakage is pA–nA scale) |

### Step 7 — Save to CSV

Each row is written to CSV incrementally (stream-write), so no pandas is required. The final file has a header + 10,000 data rows = **10,001 lines**.

---

## Validation Assertions

The following conditions are verified before simulation:

```python
assert df["vdd"].between(0.81, 0.99).all()
assert df["temp"].between(-40, 125).all()
assert df["w"].between(45e-9, 360e-9).all()
assert (df["vd"] >= 0).all() and (df["vd"] <= df["vdd"]).all()
assert (df["vs"] >= 0).all() and (df["vs"] <= df["vdd"]).all()

# PMOS ON:  vg = 0,   vb = vdd
# PMOS OFF: vg = vdd, vb = vdd

for name, nominal_val in nominal.items():
    lo = min(0.9 * nominal_val, 1.1 * nominal_val)
    hi = max(0.9 * nominal_val, 1.1 * nominal_val)
    assert df[name].between(lo, hi).all()
```

---

## Comparison: NMOS vs PMOS Datasets

| Property | NMOS | PMOS |
|----------|------|------|
| Model type | `nmos` (level 54) | `pmos` (level 54) |
| `vth0` nominal | +0.3423 V | **−0.23122 V** |
| `toxe` | 9.0 × 10⁻¹⁰ m | 9.2 × 10⁻¹⁰ m |
| `ndep` | 6.5 × 10¹⁸ cm⁻³ | 2.8 × 10¹⁸ cm⁻³ |
| `u0` | 0.02947 | **0.00391** (~7.5× lower) |
| `vb` | 0 V (GND) | `vdd` (VDD rail) |
| ON → `vg` | `vdd` | **0 V** |
| OFF → `vg` | 0 V | **`vdd`** |
| Seed | 12175305 | **12175305** (same) |
| Columns | 16 | 16 (identical schema) |

---

## Reproducibility

To regenerate the datasets from scratch:

```bash
cd DVD-Project/
python3 Data_Generation/Scripts/generate_pmos_datasets.py
```

This standalone script requires only `numpy` (system-available) and Python stdlib modules (`re`, `csv`, `subprocess`, `os`). No pandas or external packages needed.

> The seed `12175305` is fixed in all scripts. The last two digits of the combined roll numbers were used as specified.

