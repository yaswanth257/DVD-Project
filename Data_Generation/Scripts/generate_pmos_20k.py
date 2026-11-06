"""
PMOS 20k dataset generator — same seed as the 10k run.

Strategy
--------
The original simulate_pmos_on.py / simulate_pmos_off.py each call
generate_pmos_on_inputs() / generate_pmos_off_inputs(), which both
create a fresh np.random.default_rng(SEED=12175305) and draw 10 000
samples in this exact order:
  vdd (10k) → vd (10k) → vs (10k) → temp (10k) → w (10k)
  → vth0 (10k) → toxe (10k) → ndep (10k) → u0 (10k) → xj (10k)

To extend to 20k while keeping rows 1–10 000 identical to the existing
CSVs, we replay the same RNG call sequence TWICE:
  - First pass  draws 10k → reproduces the original data bit-for-bit
  - Second pass draws another 10k using the continuing RNG state

Each mode (ON / OFF) gets its own independent fresh RNG, matching the
original design.  The existing 10k CSVs are kept untouched.

Outputs
-------
  Data_Generation/Datasets/PMOS_ON_20k.csv   — 20 000 rows  (ON  mode)
  Data_Generation/Datasets/PMOS_OFF_20k.csv  — 20 000 rows  (OFF mode)

Run from the project root:
    python3 Data_Generation/Scripts/generate_pmos_20k.py
"""

import csv
import os
import re
import subprocess

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────
SEED        = 12175305        # same seed as original 10k run
N_OLD       = 10_000          # size of the original datasets (kept untouched)
N_NEW       = 10_000          # additional new samples to generate
NUM_SAMPLES = N_OLD + N_NEW   # 20 000 total

NOMINAL_VDD = 0.9
LMIN        = 45e-9

VDD_MIN = NOMINAL_VDD * 0.9   # 0.81 V
VDD_MAX = NOMINAL_VDD * 1.1   # 0.99 V

MODEL_FILE  = "45nm_MGK.pm"
NGSPICE_CMD = "ngspice_con"   # console ngspice on this system

NETLIST_TMP_ON  = "Data_Generation/Scripts/PMOS_ON_20k_tmp.cir"
NETLIST_TMP_OFF = "Data_Generation/Scripts/PMOS_OFF_20k_tmp.cir"

OUT_ON  = "Data_Generation/Datasets/PMOS_ON_20k.csv"
OUT_OFF = "Data_Generation/Datasets/PMOS_OFF_20k.csv"

COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
           "vth0", "toxe", "ndep", "u0", "xj",
           "id", "ig", "is", "ib"]


# ── Extract PMOS nominal parameters from model file ───────────────────────────
def extract_pmos_parameters(model_file):
    with open(model_file, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"(?ims)^\s*\.model\s+pmos\b(.*)", text)
    if match is None:
        raise ValueError("PMOS model section not found.")

    section = match.group(1)
    names   = ["vth0", "toxe", "ndep", "u0", "xj"]
    nominal = {}

    for name in names:
        m = re.search(
            rf"\b{name}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            section,
            re.IGNORECASE
        )
        if m is None:
            raise ValueError(f"Parameter '{name}' not found in PMOS section.")
        nominal[name] = float(m.group(1))

    return nominal


# ── Generate 20k input arrays preserving exact RNG continuity ────────────────
def generate_inputs(nominal):
    """
    Replay the EXACT original 10k RNG call sequence, then continue drawing
    another 10k with the same RNG state.  This guarantees:
      - Rows 0–9999  are bit-for-bit identical to the existing 10k CSVs.
      - Rows 10000–19999 are the new samples from the extended seed sequence.

    Draw order (matches original generate_pmos_on/off_inputs.py exactly):
      vdd_old → vd_old → vs_old → temp_old → w_old
      → vth0_old → toxe_old → ndep_old → u0_old → xj_old   (10k each)
      vdd_new → vd_new → vs_new → temp_new → w_new
      → vth0_new → toxe_new → ndep_new → u0_new → xj_new   (10k each)
    """
    rng = np.random.default_rng(SEED)

    # ── First 10k (identical to original) ──
    vdd_old  = rng.uniform(VDD_MIN, VDD_MAX, N_OLD)
    vd_old   = rng.uniform(0, vdd_old)
    vs_old   = rng.uniform(0, vdd_old)
    temp_old = rng.uniform(-40, 125, N_OLD)
    w_old    = rng.uniform(LMIN, 8 * LMIN, N_OLD)

    proc_old = {}
    for name, val in nominal.items():
        lo = min(0.9 * val, 1.1 * val)
        hi = max(0.9 * val, 1.1 * val)
        proc_old[name] = rng.uniform(lo, hi, N_OLD)

    # ── Next 10k (new samples, RNG continues) ──
    vdd_new  = rng.uniform(VDD_MIN, VDD_MAX, N_NEW)
    vd_new   = rng.uniform(0, vdd_new)
    vs_new   = rng.uniform(0, vdd_new)
    temp_new = rng.uniform(-40, 125, N_NEW)
    w_new    = rng.uniform(LMIN, 8 * LMIN, N_NEW)

    proc_new = {}
    for name, val in nominal.items():
        lo = min(0.9 * val, 1.1 * val)
        hi = max(0.9 * val, 1.1 * val)
        proc_new[name] = rng.uniform(lo, hi, N_NEW)

    # ── Concatenate into 20k arrays ──
    vdd  = np.concatenate([vdd_old,  vdd_new])
    vd   = np.concatenate([vd_old,   vd_new])
    vs   = np.concatenate([vs_old,   vs_new])
    temp = np.concatenate([temp_old, temp_new])
    w    = np.concatenate([w_old,    w_new])

    process = {}
    for name in nominal:
        process[name] = np.concatenate([proc_old[name], proc_new[name]])

    return dict(vdd=vdd, vd=vd, vs=vs, temp=temp, w=w, process=process)


# ── ngspice helpers ───────────────────────────────────────────────────────────
def fmt(v):
    return f"{float(v):.16g}"


def run_simulation(row, netlist_tmp):
    netlist = f"""\
* PMOS single-point OP

.include "{MODEL_FILE}"

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9

.temp {fmt(row['temp'])}

M1 drain gate source body pmos W={fmt(row['w'])} L={LMIN}

Vd drain 0 DC {fmt(row['vd'])}
Vg gate   0 DC {fmt(row['vg'])}
Vs source 0 DC {fmt(row['vs'])}
Vb body   0 DC {fmt(row['vb'])}

.control
set numdgt=15

altermod pmos vth0 = {fmt(row['vth0'])}
altermod pmos toxe = {fmt(row['toxe'])}
altermod pmos ndep = {fmt(row['ndep'])}
altermod pmos u0   = {fmt(row['u0'])}
altermod pmos xj   = {fmt(row['xj'])}

op

let id_val = i(Vd)
let ig_val = i(Vg)
let is_val = i(Vs)
let ib_val = i(Vb)

echo RESULT $&id_val $&ig_val $&is_val $&ib_val

.endc
.end
"""
    with open(netlist_tmp, "w", encoding="utf-8") as f:
        f.write(netlist)

    try:
        result = subprocess.run(
            [NGSPICE_CMD, "-b", netlist_tmp],
            capture_output=True, text=True, errors="replace", check=False
        )
        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(f"ngspice failed:\n{output[:800]}")

        m = re.search(
            r"RESULT\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            output
        )
        if m is None:
            raise RuntimeError(f"Cannot parse RESULT:\n{output[:800]}")

        return [float(x) for x in m.groups()]

    finally:
        if os.path.exists(netlist_tmp):
            os.remove(netlist_tmp)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(mode, output_file, netlist_tmp, nominal):
    """
    mode        : 'ON' or 'OFF'
    output_file : destination CSV
    nominal     : dict of PMOS nominal parameters

    Each call gets its own fresh generate_inputs() (fresh RNG from SEED),
    matching the original one-RNG-per-mode design.
    """
    base = generate_inputs(nominal)   # fresh RNG(SEED) each call

    vdd  = base["vdd"]
    vd   = base["vd"]
    vs   = base["vs"]
    proc = base["process"]

    if mode == "ON":
        vg = np.zeros(NUM_SAMPLES)
        vb = vdd.copy()
    else:   # OFF
        vg = vdd.copy()
        vb = vdd.copy()

    print(f"\n{'='*60}")
    print(f"Starting PMOS_{mode} 20k simulations: {NUM_SAMPLES} samples total")
    print(f"  → Rows      1–{N_OLD:5d}  reproduce the original 10k (same seed)")
    print(f"  → Rows {N_OLD+1:5d}–{NUM_SAMPLES:5d}  are NEW (seed continues)")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(COLUMNS)

        kcl_max = 0.0

        for i in range(NUM_SAMPLES):
            row = {
                "vdd":  vdd[i],
                "temp": base["temp"][i],
                "w":    base["w"][i],
                "vd":   vd[i],
                "vs":   vs[i],
                "vg":   vg[i],
                "vb":   vb[i],
                "vth0": proc["vth0"][i],
                "toxe": proc["toxe"][i],
                "ndep": proc["ndep"][i],
                "u0":   proc["u0"][i],
                "xj":   proc["xj"][i],
            }

            id_, ig, is_, ib = run_simulation(row, netlist_tmp)

            kcl = abs(id_ + ig + is_ + ib)
            if kcl > kcl_max:
                kcl_max = kcl

            writer.writerow([
                row["vdd"], row["temp"], row["w"],
                row["vd"],  row["vs"],  row["vg"],  row["vb"],
                row["vth0"], row["toxe"], row["ndep"], row["u0"], row["xj"],
                id_, ig, is_, ib
            ])

            if (i + 1) % 500 == 0 or i == 0:
                label = "NEW" if i >= N_OLD else "   "
                pct   = (i + 1) / NUM_SAMPLES * 100
                print(f"  {label} [{i+1:5d}/{NUM_SAMPLES}]  {pct:5.1f}%  "
                      f"KCL_max = {kcl_max:.4e} A", flush=True)

    print(f"\nDone! Dataset saved: {output_file}")
    print(f"Maximum absolute KCL residual: {kcl_max:.6e} A")


def main():
    print("Reading PMOS parameters from model file...")
    nominal = extract_pmos_parameters(MODEL_FILE)
    print("Nominal PMOS parameters:")
    for k, v in nominal.items():
        print(f"  {k:6s} = {v}")

    # Each mode uses its own independent RNG (fresh from SEED) — matching the
    # original one-RNG-per-mode design.
    run_pipeline("ON",  OUT_ON,  NETLIST_TMP_ON,  nominal)
    run_pipeline("OFF", OUT_OFF, NETLIST_TMP_OFF, nominal)

    print("\nAll PMOS 20k datasets generated successfully!")
    print(f"  {OUT_ON}")
    print(f"  {OUT_OFF}")
    print(f"\nOriginal 10k files are untouched:")
    print(f"  Data_Generation/Datasets/PMOS_ON.csv")
    print(f"  Data_Generation/Datasets/PMOS_OFF.csv")


if __name__ == "__main__":
    main()
