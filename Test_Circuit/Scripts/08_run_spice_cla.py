"""Step 8 - SPICE reference for the CLA: all 128 input combinations at all 200 corners.

One ngspice run per corner (128 input combinations via alter + op in a single run).
Leakage definition (same as Part B):
    leak_vdd   = current delivered by the VDD source (positive = from supply)
    leak_total = VDD source + every primary-input source sitting at logic 1

Usage (from repository root):
    python3 Test_Circuit/Scripts/08_run_spice_cla.py
"""
import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Gate_Validation" / "Scripts"))

import pandas as pd
from circuit_common import (CORNERS_CSV, INPUTS, NGSPICE, SPICE_CSV, all_input_combos,
                             combo_label, fmt)
from gate_common import LMIN, MODEL_FILE, OPTIONS, WN_INV, WN_NAND, WP_INV, WP_NAND, WN_NOR, WP_NOR, PROC
from gate_common import spice_leak as _gate_spice_leak
from circuit_common import GATES


# ===========================================================================
# Transistor expansion (same as 07_write_cla_netlist.py)
# ===========================================================================

def _transistors_for_gate(inst_name, gate_type, in_nets, out_net, vdd_net="vdd"):
    lines = []
    pfx = inst_name.replace("-", "_")

    def tline(nm, typ, d, g, s, w):
        body = vdd_net if typ == "pmos" else "0"
        return f"M{pfx}_{nm} {d} {g} {s} {body} {typ} W={fmt(w)} L={fmt(LMIN)}"

    if gate_type == "NAND2":
        A, B = in_nets[0], in_nets[1]
        x = f"{pfx}_x"
        lines += [tline("mp1", "pmos", out_net, A, vdd_net, WP_NAND),
                  tline("mp2", "pmos", out_net, B, vdd_net, WP_NAND),
                  tline("mn1", "nmos", out_net, A, x,       WN_NAND),
                  tline("mn2", "nmos", x,       B, "0",     WN_NAND)]
    elif gate_type == "AND2":
        A, B = in_nets[0], in_nets[1]
        x, n = f"{pfx}_x", f"{pfx}_n"
        lines += [tline("mp1", "pmos", n, A, vdd_net, WP_NAND),
                  tline("mp2", "pmos", n, B, vdd_net, WP_NAND),
                  tline("mn1", "nmos", n, A, x,       WN_NAND),
                  tline("mn2", "nmos", x, B, "0",     WN_NAND),
                  tline("mp3", "pmos", out_net, n, vdd_net, WP_INV),
                  tline("mn3", "nmos", out_net, n, "0",     WN_INV)]
    elif gate_type == "OR2":
        A, B = in_nets[0], in_nets[1]
        y, n = f"{pfx}_y", f"{pfx}_n"
        lines += [tline("mp1", "pmos", y,       A, vdd_net, WP_NOR),
                  tline("mp2", "pmos", n,       B, y,       WP_NOR),
                  tline("mn1", "nmos", n,       A, "0",     WN_NOR),
                  tline("mn2", "nmos", n,       B, "0",     WN_NOR),
                  tline("mp3", "pmos", out_net, n, vdd_net, WP_INV),
                  tline("mn3", "nmos", out_net, n, "0",     WN_INV)]
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")
    return lines


def build_spice_netlist(corner):
    """Full CLA netlist for one corner, all 128 input combos via alter + op."""
    vdd  = corner["vdd"]
    combos = all_input_combos()
    first  = combos[0]

    lines = [
        "* CLA3 leakage sweep: all 128 input combinations at one PVT corner",
        f"* corner {int(corner['corner'])}: VDD = {fmt(vdd)} V, T = {fmt(corner['temp'])} C",
        f'.include "{MODEL_FILE}"',
        "",
        OPTIONS,
        "",
        f".temp {fmt(corner['temp'])}",
        "",
        f"Vdd vdd 0 DC {fmt(vdd)}",
    ]
    for i, name in enumerate(INPUTS):
        lines.append(f"V{name} {name} 0 DC {fmt(first[i] * vdd)}")
    lines.append("")

    for (inst, gate_type, in_nets, out_net) in GATES:
        lines.extend(_transistors_for_gate(inst, gate_type, in_nets, out_net))

    lines += ["", ".control", "set numdgt=15", ""]
    for dev in ("nmos", "pmos"):
        for pname in PROC:
            lines.append(f"altermod {dev} {pname} = {fmt(corner[dev[0] + '_' + pname])}")
    lines.append("")

    # Probe string: i(Vdd) then all input sources
    probes = ["i(Vdd)"] + [f"i(V{n})" for n in INPUTS]

    for combo in combos:
        for i, name in enumerate(INPUTS):
            lines.append(f"alter V{name} = {fmt(combo[i] * vdd)}")
        lines.append("op")
        lines.append("echo RESULT " + combo_label(combo) + " " + " ".join(f"$&{p}" for p in probes))

    lines += ["", ".endc", ".end", ""]
    return "\n".join(lines)


def run_one(args):
    corner, = args if isinstance(args, (list, tuple)) and len(args) == 1 else (args,)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cla.cir"
        path.write_text(build_spice_netlist(corner))
        result = subprocess.run([NGSPICE, "-b", str(path)],
                                capture_output=True, text=True, errors="replace")
        out = result.stdout

    res = {}
    for line in out.splitlines():
        if line.startswith("RESULT"):
            parts = line.split()
            res[parts[1]] = [float(x) for x in parts[2:]]

    rows = []
    for combo in all_input_combos():
        lab = combo_label(combo)
        if lab not in res:
            raise RuntimeError(f"Corner {int(corner['corner'])} input {lab}: no RESULT\n{out[-800:]}")
        vals = res[lab]
        i_vdd   = vals[0]
        i_inputs = vals[1:]
        leak_vdd, leak_total = _gate_spice_leak(i_vdd, i_inputs, combo)
        a0, a1, a2, b0, b1, b2, cin = combo
        rows.append({
            "corner":          int(corner["corner"]),
            "vdd":             corner["vdd"],
            "temp":            corner["temp"],
            "input_combo":     lab,
            "a0": a0, "a1": a1, "a2": a2,
            "b0": b0, "b1": b1, "b2": b2,
            "cin":             cin,
            "spice_leak_vdd":  leak_vdd,
            "spice_leak_total": leak_total,
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    corners = pd.read_csv(CORNERS_CSV).to_dict("records")
    print(f"Running {len(corners)} corners x 128 input combos with {args.workers} workers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [r for rows in ex.map(run_one, corners) for r in rows]

    df = pd.DataFrame(results)
    SPICE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPICE_CSV, index=False)
    print(f"{len(df)} SPICE operating points -> {SPICE_CSV}")
    print("Median spice_leak_total:", df["spice_leak_total"].median())
    print("Non-positive rows:", int((df["spice_leak_total"] <= 0).sum()))
