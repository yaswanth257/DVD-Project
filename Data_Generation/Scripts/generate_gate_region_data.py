"""Extra training samples for the ON models in the region where ON transistors operate inside gates.

    python3 Data_Generation/Scripts/generate_gate_region_data.py        (from the repository root, about 15 s with 6 processes)

Inside a static CMOS gate an ON transistor connects a node to a rail: both terminals sit near the same rail and |Vds| is only
20 uV - 1 mV (or a threshold voltage, for a "degraded" level). Uniform sampling of vd and vs (section 4) almost never produces
such points (|Vds| < 1 mV in 0.2 % of the rows), so the models' gate and body currents were 2 - 7 % off there. This script draws
4,000 extra rows per ON case with everything as in section 4 (VDD, T, W, process +-10 %), except vd / vs:
    vs : 1/3 near 0 (vs = 10^U(-7, log10 0.45) V), 1/3 near VDD (VDD - the same), 1/3 uniform in [0, VDD]
    vd : 1/2 on the diagonal (vd = vs +- 10^U(-7, 0) V), 1/2 drawn like vs (rail corners and edges); mirrored into [0, VDD]
It uses the same single-transistor netlist as the main datasets and no gate simulation result. Output, same columns as the
other datasets: Datasets/NMOS_ON_gate_region.csv, Datasets/PMOS_ON_gate_region.csv. The training script
(ML_Models/Scripts/train_on_models_gate_region.py) applies the data-check rule (no zero current, Kirchhoff residual <= 1e-3 of the
largest current), which removes about 2 % of the NMOS_ON rows: transistors with all terminals within microvolts of each other,
whose currents are at the solver's noise floor.
"""
import os
import re
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "45nm_MGK.pm"
OUT_DIR = ROOT / "Data_Generation" / "Datasets"
NGSPICE_CMD = shutil.which("ngspice_con") or "ngspice"
SEED, NOMINAL_VDD, LMIN, ROWS = 12175305, 0.9, 45e-9, 4000
PROC = ["vth0", "toxe", "ndep", "u0", "xj"]
CASES = {"NMOS_ON": ("nmos", 1, 0, SEED + 1000), "PMOS_ON": ("pmos", 0, 1, SEED + 1002)}   # type, vg / VDD, vb / VDD, generator seed
COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROC, "id", "ig", "is", "ib"]


def fmt(v):
    return f"{float(v):.16g}"


def nominals():
    text = MODEL_FILE.read_text()
    out = {}
    for dev in ("nmos", "pmos"):
        block = re.split(r"\.model\s+" + dev + r"\s", text, flags=re.I)[1]
        block = re.split(r"\.model\s", block, flags=re.I)[0]
        out[dev] = {p: float(re.search(r"\b" + p + r"\s*=\s*([-+0-9.eE]+)", block).group(1)) for p in PROC}
    return out


def terminal(rng, vdd):
    n = len(vdd)
    kind = rng.integers(0, 3, n)
    offset = 10 ** rng.uniform(-7, np.log10(0.45), n)
    return np.where(kind == 0, offset, np.where(kind == 1, vdd - offset, rng.uniform(0, vdd)))


def mirror(v, vdd):
    v = np.where(v < 0, -v, v)
    return np.where(v > vdd, 2 * vdd - v, v)


def draw(typ, n, rng, nom):
    vdd = rng.uniform(0.9 * NOMINAL_VDD, 1.1 * NOMINAL_VDD, n)
    df = pd.DataFrame({"vdd": vdd, "temp": rng.uniform(-40, 125, n), "w": rng.uniform(LMIN, 8 * LMIN, n)})
    vs = terminal(rng, vdd)
    diagonal = rng.random(n) < 0.5
    vd_diagonal = vs + rng.choice([-1.0, 1.0], n) * 10 ** rng.uniform(-7, 0, n)
    vd = np.where(diagonal, vd_diagonal, terminal(rng, vdd))
    df["vd"], df["vs"] = np.clip(mirror(vd, vdd), 0, vdd), vs
    for p in PROC:
        lo, hi = sorted((0.9 * nom[typ][p], 1.1 * nom[typ][p]))
        df[p] = rng.uniform(lo, hi, n)
    return df


def simulate(args):
    """One operating point with the single-transistor netlist of the main datasets."""
    typ, row = args
    alter = "\n".join(f"altermod {typ} {p} = {fmt(row[p])}" for p in PROC)
    netlist = f"""* {typ.upper()} single-point OP
.include "{MODEL_FILE}"
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.temp {fmt(row['temp'])}
M1 drain gate source body {typ} W={fmt(row['w'])} L={LMIN}
Vd drain 0 DC {fmt(row['vd'])}
Vg gate   0 DC {fmt(row['vg'])}
Vs source 0 DC {fmt(row['vs'])}
Vb body   0 DC {fmt(row['vb'])}
.control
set numdgt=15
{alter}
op
let id_val = i(Vd)
let ig_val = i(Vg)
let is_val = i(Vs)
let ib_val = i(Vb)
echo RESULT $&id_val $&ig_val $&is_val $&ib_val
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        path = f.name
    try:
        out = subprocess.run([NGSPICE_CMD, "-b", path], capture_output=True, text=True, errors="replace").stdout
    finally:
        os.remove(path)
    m = re.search(r"RESULT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", out)
    if m is None:
        raise RuntimeError(f"no RESULT line from ngspice:\n{out[:800]}")
    return [float(x) for x in m.groups()]


if __name__ == "__main__":
    nom = nominals()
    with Pool(6) as pool:
        for case, (typ, g_on, b_on, seed) in CASES.items():
            df = draw(typ, ROWS, np.random.default_rng(seed), nom)
            df["vg"], df["vb"] = g_on * df["vdd"], b_on * df["vdd"]
            currents = np.array(pool.map(simulate, [(typ, r) for _, r in df.iterrows()], chunksize=50))
            df[["id", "ig", "is", "ib"]] = currents
            df = df[COLUMNS]
            df.to_csv(OUT_DIR / f"{case}_gate_region.csv", index=False)
            kcl = np.abs(currents.sum(1)) / np.abs(currents).max(1)
            ok = ~(currents == 0).any(1) & (kcl <= 1e-3)
            print(f"{case}: {len(df)} rows -> {(OUT_DIR / f'{case}_gate_region.csv').relative_to(ROOT)}; |vd - vs| < 1 mV in "
                  f"{np.mean(np.abs(df.vd - df.vs) < 1e-3):.0%}; pass the data-check rule: {ok.sum()} ({ok.mean():.1%})")
