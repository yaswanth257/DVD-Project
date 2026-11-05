"""
Standalone PMOS ON + OFF dataset generator + ngspice simulator.
No pandas required — uses only numpy, csv, re, subprocess (all available).

Usage (run from project root):
    python3 Data_Generation/Scripts/generate_pmos_datasets.py

Produces:
    Data_Generation/Datasets/PMOS_ON.csv
    Data_Generation/Datasets/PMOS_OFF.csv
"""

import csv
import os
import re
import subprocess
import sys

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────
SEED        = 12175305
NUM_SAMPLES = 10000

NOMINAL_VDD = 0.9
LMIN        = 45e-9

VDD_MIN = NOMINAL_VDD * 0.9   # 0.81 V
VDD_MAX = NOMINAL_VDD * 1.1   # 0.99 V

MODEL_FILE     = "45nm_MGK.pm"
NGSPICE_CMD    = "ngspice"          # batch ngspice on this system
NETLIST_TMP    = "/tmp/pmos_tmp.cir"

OUT_ON  = "Data_Generation/Datasets/PMOS_ON.csv"
OUT_OFF = "Data_Generation/Datasets/PMOS_OFF.csv"

COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
           "vth0", "toxe", "ndep", "u0", "xj",
           "id", "ig", "is", "ib"]


# ── Extract PMOS nominal parameters from model file ───────────────────────────
def extract_pmos_parameters(model_file):
    with open(model_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Grab everything from ".model pmos" to end of file
    match = re.search(r"(?ims)^\s*\.model\s+pmos\b(.*)", text)
    if match is None:
        raise ValueError("PMOS model section not found.")

    section = match.group(1)

    names = ["vth0", "toxe", "ndep", "u0", "xj"]
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


# ── Generate input arrays ────────────────────────────────────────────────────
def generate_inputs(nominal):
    """Return a dict of numpy arrays (10 000 rows each)."""
    rng = np.random.default_rng(SEED)

    vdd  = rng.uniform(VDD_MIN, VDD_MAX, NUM_SAMPLES)
    vd   = rng.uniform(0, vdd)         # element-wise upper bound = vdd
    vs   = rng.uniform(0, vdd)
    temp = rng.uniform(-40, 125, NUM_SAMPLES)
    w    = rng.uniform(LMIN, 8 * LMIN, NUM_SAMPLES)

    process = {}
    for name, val in nominal.items():
        lo = min(0.9 * val, 1.1 * val)
        hi = max(0.9 * val, 1.1 * val)
        process[name] = rng.uniform(lo, hi, NUM_SAMPLES)

    return dict(vdd=vdd, vd=vd, vs=vs, temp=temp, w=w, process=process)


# ── Validate arrays ──────────────────────────────────────────────────────────
def validate_inputs(arrs, nominal, mode):
    vdd  = arrs["vdd"]
    vd   = arrs["vd"]
    vs   = arrs["vs"]
    temp = arrs["temp"]
    w    = arrs["w"]
    vg   = arrs["vg"]
    vb   = arrs["vb"]
    proc = arrs["process"]

    assert len(vdd) == NUM_SAMPLES
    assert np.all((vdd >= VDD_MIN) & (vdd <= VDD_MAX))
    assert np.all((temp >= -40) & (temp <= 125))
    assert np.all((w >= LMIN) & (w <= 8 * LMIN))
    assert np.all(vd >= 0) and np.all(vd <= vdd)
    assert np.all(vs >= 0) and np.all(vs <= vdd)

    if mode == "ON":
        assert np.all(vg == 0),    "PMOS ON: vg must be 0 V"
        assert np.all(vb == vdd),  "PMOS ON: vb must equal vdd"
    elif mode == "OFF":
        assert np.all(vg == vdd),  "PMOS OFF: vg must equal vdd"
        assert np.all(vb == vdd),  "PMOS OFF: vb must equal vdd"

    for name, val in nominal.items():
        lo = min(0.9 * val, 1.1 * val)
        hi = max(0.9 * val, 1.1 * val)
        arr = proc[name]
        assert np.all((arr >= lo) & (arr <= hi)), f"{name} out of ±10% range"


# ── ngspice netlist & simulation ─────────────────────────────────────────────
def fmt(v):
    return f"{float(v):.16g}"


def run_simulation(row):
    """
    row: dict with keys vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj
    Returns [id, ig, is_, ib] floats.
    """
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
    with open(NETLIST_TMP, "w", encoding="utf-8") as f:
        f.write(netlist)

    try:
        result = subprocess.run(
            [NGSPICE_CMD, "-b", NETLIST_TMP],
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
        if os.path.exists(NETLIST_TMP):
            os.remove(NETLIST_TMP)


# ── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(mode, output_file, nominal):
    """mode: 'ON' or 'OFF'"""
    base = generate_inputs(nominal)

    vdd = base["vdd"]
    vd  = base["vd"]
    vs  = base["vs"]

    if mode == "ON":
        vg = np.zeros(NUM_SAMPLES)
        vb = vdd.copy()
    else:
        vg = vdd.copy()
        vb = vdd.copy()

    arrs = dict(**base, vg=vg, vb=vb)
    validate_inputs(arrs, nominal, mode)

    proc = base["process"]

    print(f"\n{'='*60}")
    print(f"Starting PMOS_{mode} simulations: {NUM_SAMPLES} samples")
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

            id_, ig, is_, ib = run_simulation(row)

            kcl = abs(id_ + ig + is_ + ib)
            if kcl > kcl_max:
                kcl_max = kcl

            writer.writerow([
                row["vdd"], row["temp"], row["w"],
                row["vd"],  row["vs"],   row["vg"],  row["vb"],
                row["vth0"], row["toxe"], row["ndep"], row["u0"], row["xj"],
                id_, ig, is_, ib
            ])

            if (i + 1) % 100 == 0 or i == 0:
                pct = (i + 1) / NUM_SAMPLES * 100
                print(f"  [{i+1:5d}/{NUM_SAMPLES}]  {pct:5.1f}%  "
                      f"KCL_max = {kcl_max:.4e} A", flush=True)

    print(f"\nDone! Dataset saved: {output_file}")
    print(f"Maximum absolute KCL residual: {kcl_max:.6e} A")


def main():
    print("Reading PMOS parameters from model file...")
    nominal = extract_pmos_parameters(MODEL_FILE)
    print("Nominal PMOS parameters:")
    for k, v in nominal.items():
        print(f"  {k:6s} = {v}")

    # Always generate BOTH datasets from the same seed / same RNG sequence
    run_pipeline("ON",  OUT_ON,  nominal)
    run_pipeline("OFF", OUT_OFF, nominal)

    print("\nAll PMOS datasets generated successfully!")


if __name__ == "__main__":
    main()
