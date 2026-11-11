"""Step 3 - SPICE reference: every gate, every input combination, every corner (18 cases x 200 corners = 3,600 operating points).
One ngspice run per gate and corner (all input combinations via alter + op). Records both leakage definitions, the internal
node voltages and the output voltage. SPICE results are for evaluation only; nothing here is used for training.
"""
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from gate_common import (CORNERS_CSV, GATE_ORDER, GATES, MODEL_FILE, NGSPICE, SPICE_CSV, input_combos, netlist,
                         combo_label, spice_leak)


def run_one(args):
    gate, corner = args
    g = GATES[gate]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{gate}.cir"
        path.write_text(netlist(gate, corner, str(MODEL_FILE)))
        out = subprocess.run([NGSPICE, "-b", str(path)], capture_output=True, text=True, errors="replace").stdout
    rows = []
    res = {l.split()[1]: l.split()[2:] for l in out.splitlines() if l.startswith("RESULT")}
    k = len(g["inputs"])
    for combo in input_combos(gate):
        lab = combo_label(combo)
        if lab not in res:
            raise RuntimeError(f"{gate} corner {int(corner['corner'])} input {lab}: no RESULT\n{out[-800:]}")
        vals = [float(x) for x in res[lab]]
        i_vdd, i_in, volts = vals[0], vals[1:1 + k], vals[1 + k:]
        leak_vdd, leak_total = spice_leak(i_vdd, i_in, combo)
        row = {"gate": gate, "input": lab, "corner": int(corner["corner"]), "vdd": corner["vdd"], "temp": corner["temp"],
               "spice_leak_vdd": leak_vdd, "spice_leak_total": leak_total}
        for name, v in zip(g["nodes"] + [g["out"]], volts):
            row[f"spice_v_{name}"] = v
        rows.append(row)
    return rows


if __name__ == "__main__":
    corners = pd.read_csv(CORNERS_CSV).to_dict("records")
    jobs = [(g, c) for g in GATE_ORDER for c in corners]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = [r for rows in ex.map(run_one, jobs) for r in rows]
    df = pd.DataFrame(results)
    df.to_csv(SPICE_CSV, index=False)
    print(f"{len(df)} SPICE operating points -> {SPICE_CSV}")
    print(df.groupby(["gate", "input"])[["spice_leak_vdd", "spice_leak_total"]].median().to_string())
    print("non-positive leakage rows:", int((df["spice_leak_total"] <= 0).sum()))
