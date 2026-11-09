"""Write the example single-transistor netlists (deliverable 2) and check them against the datasets.

    python3 Data_Generation/Scripts/write_example_netlists.py        (from the repository root)

For each case (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) the netlist of row 1 of the dataset is written to
Data_Generation/Netlists/<case>.cir. The netlist text is the one the generators simulate for every sample
(`run_simulation` in generate_nmos_20k.py / generate_pmos_20k.py), only the model path is written relative to the netlist
(ngspice resolves `.include` relative to the netlist file, so it runs from any directory). Each netlist is then simulated
with ngspice and its four currents are compared with the row in the CSV.

Run one by hand:   ngspice -b Data_Generation/Netlists/NMOS_OFF.cir     (prints RESULT id ig is ib)
"""
import os
import re
import shutil
import subprocess

import numpy as np

CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
DATA_DIR = "Data_Generation/Datasets"
OUT_DIR = "Data_Generation/Netlists"
MODEL_INCLUDE = "../../45nm_MGK.pm"          # relative to Data_Generation/Netlists/
LMIN = 45e-9
NGSPICE_CMD = shutil.which("ngspice_con") or "ngspice"
COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]


def fmt(v):
    return f"{float(v):.16g}"


def netlist(case, row):
    dev = case.split("_")[0].lower()           # nmos / pmos
    return f"""\
* {dev.upper()} single-point OP
* Example netlist for {case}: row 1 of Data_Generation/Datasets/{case}.csv (= row 1 of {case}_20k.csv)
* vdd = {fmt(row['vdd'])} V; SPICE currents in the dataset (A, positive = out of the transistor at that terminal):
*   id = {row['id']:.6e}   ig = {row['ig']:.6e}   is = {row['is']:.6e}   ib = {row['ib']:.6e}
* Every sample of the dataset is this netlist with that sample's values.

.include "{MODEL_INCLUDE}"

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9

.temp {fmt(row['temp'])}

M1 drain gate source body {dev} W={fmt(row['w'])} L={LMIN}

Vd drain 0 DC {fmt(row['vd'])}
Vg gate   0 DC {fmt(row['vg'])}
Vs source 0 DC {fmt(row['vs'])}
Vb body   0 DC {fmt(row['vb'])}

.control
set numdgt=15

altermod {dev} vth0 = {fmt(row['vth0'])}
altermod {dev} toxe = {fmt(row['toxe'])}
altermod {dev} ndep = {fmt(row['ndep'])}
altermod {dev} u0   = {fmt(row['u0'])}
altermod {dev} xj   = {fmt(row['xj'])}

op

let id_val = i(Vd)
let ig_val = i(Vg)
let is_val = i(Vs)
let ib_val = i(Vb)

echo RESULT $&id_val $&ig_val $&is_val $&ib_val

.endc
.end
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    worst_all = 0.0
    for case in CASES:
        data = np.loadtxt(f"{DATA_DIR}/{case}.csv", delimiter=",", skiprows=1, max_rows=1)
        row = dict(zip(COLUMNS, data))
        path = f"{OUT_DIR}/{case}.cir"
        with open(path, "w", encoding="utf-8") as f:
            f.write(netlist(case, row))
        out = subprocess.run([NGSPICE_CMD, "-b", path], capture_output=True, text=True, errors="replace").stdout
        m = re.search(r"RESULT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", out)
        if m is None:
            raise RuntimeError(f"{case}: no RESULT line from ngspice:\n{out[:800]}")
        sim = np.array([float(x) for x in m.groups()])
        ref = np.array([row[c] for c in ("id", "ig", "is", "ib")])
        worst = float(np.max(np.abs(sim - ref) / np.abs(ref)))
        worst_all = max(worst_all, worst)
        print(f"{path:38s} ngspice: " + "  ".join(f"{c} {v:+.6e}" for c, v in zip(("id", "ig", "is", "ib"), sim))
              + f"   worst relative difference to the CSV {worst:.1e}")
    print(f"all four netlists written; worst relative difference {worst_all:.1e}")


if __name__ == "__main__":
    main()
