"""Step 7 (fix for the gate-level error) - extra single-transistor training data at small |Vds|.

Gate transistors mostly sit at |Vds| ~ 0 (ON devices, stack nodes, both terminals at a rail), which uniform sampling of vd and vs
(Section 4) almost never produces; step 6 shows the models are 6-24 % off there but 0.1-0.7 % off elsewhere.
This script draws the same parameters as Section 4, except that |vd - vs| is log-uniform in [1e-6, 1] V (random sign, vs uniform,
both clipped to [0, VDD]). Output: Data_Generation/Datasets/<CASE>_lowvds.csv (same columns as the main datasets). Append these rows
to the training set (keep the original validation/test rows), retrain, then re-run steps 4-5.

    python3 Gate_Validation/Scripts/07_generate_low_vds_data.py [rows_per_case]      (default 5000, ~20 ms per row)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from gate_common import LMIN, MODEL_FILE, NGSPICE, NOMINAL_VDD, OPTIONS, PROC, ROOT, SEED, fmt, read_nominals

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
CASES = {"NMOS_ON": ("nmos", 1, 0), "NMOS_OFF": ("nmos", 0, 0), "PMOS_ON": ("pmos", 0, 1), "PMOS_OFF": ("pmos", 1, 1)}  # vg, vb in units of VDD
nom = read_nominals()


def draw(typ, n, rng):
    vdd = rng.uniform(0.9 * NOMINAL_VDD, 1.1 * NOMINAL_VDD, n)
    vds = 10 ** rng.uniform(-6, 0, n) * rng.choice([-1, 1], n)
    vs = rng.uniform(0, vdd)
    vd = vs + vds
    flip = (vd < 0) | (vd > vdd)
    vd = np.where(flip, vs - vds, vd)                           # mirror if the draw left [0, VDD]
    df = pd.DataFrame({"vdd": vdd, "temp": rng.uniform(-40, 125, n), "w": rng.uniform(1 * LMIN, 8 * LMIN, n),
                       "vd": np.clip(vd, 0, vdd), "vs": vs})
    for p in PROC:
        lo, hi = sorted((0.9 * nom[typ][p], 1.1 * nom[typ][p]))
        df[p] = rng.uniform(lo, hi, n)
    return df


def simulate(row, typ, vg, vb, path):
    alt = "\n".join(f"altermod {typ} {p} = {fmt(row[p])}" for p in PROC)
    path.write_text(f""".include "{MODEL_FILE}"
{OPTIONS}
.temp {fmt(row['temp'])}
M1 drain gate source body {typ} W={fmt(row['w'])} L={fmt(LMIN)}
Vd drain 0 DC {fmt(row['vd'])}
Vg gate 0 DC {fmt(vg)}
Vs source 0 DC {fmt(row['vs'])}
Vb body 0 DC {fmt(vb)}
.control
set numdgt=15
{alt}
op
echo RESULT $&i(Vd) $&i(Vg) $&i(Vs) $&i(Vb)
.endc
.end
""")
    o = subprocess.run([NGSPICE, "-b", str(path)], capture_output=True, text=True).stdout
    return [float(x) for x in [l for l in o.splitlines() if l.startswith("RESULT")][0].split()[1:]]


out_dir = ROOT / "Data_Generation" / "Datasets"
with tempfile.TemporaryDirectory() as tmp:
    for k, (case, (typ, g_on, b_on)) in enumerate(CASES.items()):
        rng = np.random.default_rng(SEED + 100 + k)
        df = draw(typ, N, rng)
        df["vg"] = g_on * df["vdd"]; df["vb"] = b_on * df["vdd"]
        cur = np.array([simulate(r, typ, r["vg"], r["vb"], Path(tmp) / "f.cir") for _, r in df.iterrows()])
        for j, c in enumerate(["id", "ig", "is", "ib"]):
            df[c] = cur[:, j]
        df = df[["vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROC, "id", "ig", "is", "ib"]]
        df.to_csv(out_dir / f"{case}_lowvds.csv", index=False)
        print(f"{case}: {len(df)} rows -> {out_dir / (case + '_lowvds.csv')}")
