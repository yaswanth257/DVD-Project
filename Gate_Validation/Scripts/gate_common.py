"""Shared definitions for Part B (gate-level validation): topology, sizing, corners, netlists, leakage definition.

Run every script from the repository root, e.g.  python3 Gate_Validation/Scripts/01_generate_corners.py
"""
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "45nm_MGK.pm"
GV = ROOT / "Gate_Validation"
CORNERS_CSV = GV / "Results" / "corners.csv"
SPICE_CSV = GV / "Results" / "gate_spice_results.csv"
ML_CSV = GV / "Results" / "gate_ml_results.csv"
NGSPICE = shutil.which("ngspice_con") or "ngspice"

SEED = 12175305
N_CORNERS = 200
NOMINAL_VDD = 0.9
LMIN = 45e-9
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
PROC = ["vth0", "toxe", "ndep", "u0", "xj"]

# ---- sizing (all W inside the trained range 1..8 x Lmin = 45..360 nm) -------------------------------------------------
WN_INV, WP_INV = 90e-9, 180e-9           # 2 Lmin / 4 Lmin
WN_NAND, WP_NAND = 180e-9, 180e-9        # series NMOS doubled
WN_NOR, WP_NOR = 90e-9, 360e-9           # series PMOS doubled (360 nm = 8 Lmin, the maximum)


# ---- topology: (name, type, drain, gate, source, W). Nets: vdd, 0 (ground), inputs A / B, internal x / y / n, out ----
def _inv(prefix, inp, out, wn=WN_INV, wp=WP_INV):
    return [(f"MP{prefix}", "pmos", out, inp, "vdd", wp), (f"MN{prefix}", "nmos", out, inp, "0", wn)]


def _nand(out):
    return [("MP1", "pmos", out, "A", "vdd", WP_NAND), ("MP2", "pmos", out, "B", "vdd", WP_NAND),
            ("MN1", "nmos", out, "A", "x", WN_NAND), ("MN2", "nmos", "x", "B", "0", WN_NAND)]


def _nor(out):
    return [("MP1", "pmos", "y", "A", "vdd", WP_NOR), ("MP2", "pmos", out, "B", "y", WP_NOR),
            ("MN1", "nmos", out, "A", "0", WN_NOR), ("MN2", "nmos", out, "B", "0", WN_NOR)]


GATES = {
    "INV":   {"inputs": ["A"],      "transistors": _inv("1", "A", "out"), "nodes": [],         "out": "out"},
    "NAND2": {"inputs": ["A", "B"], "transistors": _nand("out"),          "nodes": ["x"],      "out": "out"},
    "NOR2":  {"inputs": ["A", "B"], "transistors": _nor("out"),           "nodes": ["y"],      "out": "out"},
    # AND2 / OR2 = NAND2 / NOR2 (output net n) followed by an INV; simulated as one complete netlist
    "AND2":  {"inputs": ["A", "B"], "transistors": _nand("n") + _inv("3", "n", "out"), "nodes": ["x", "n"], "out": "out"},
    "OR2":   {"inputs": ["A", "B"], "transistors": _nor("n") + _inv("3", "n", "out"),  "nodes": ["y", "n"], "out": "out"},
}
GATE_ORDER = ["INV", "NAND2", "NOR2", "AND2", "OR2"]


def input_combos(gate):
    k = len(GATES[gate]["inputs"])
    return [tuple((c >> (k - 1 - i)) & 1 for i in range(k)) for c in range(2 ** k)]   # 00 01 10 11


def combo_label(combo):
    return "".join(str(b) for b in combo)


def case_list():
    return [(g, combo_label(c)) for g in GATE_ORDER for c in input_combos(g)]           # the 18 cases


# ---- nominal process values from the model file ------------------------------------------------------------------------
def read_nominals(model_file=MODEL_FILE):
    text = Path(model_file).read_text()
    out = {}
    for dev in ("nmos", "pmos"):
        block = re.split(r"\.model\s+" + dev + r"\s", text, flags=re.I)[1]
        block = re.split(r"\.model\s", block, flags=re.I)[0]
        vals = {}
        for name in PROC:
            m = re.search(r"\b" + name + r"\s*=\s*([-+0-9.eE]+)", block)
            vals[name] = float(m.group(1))
        out[dev] = vals
    return out


def draw_corners(n=N_CORNERS, seed=SEED):
    """One corner = VDD, temperature, one NMOS process set, one PMOS process set (section 4 / 5 of the assignment)."""
    rng = np.random.default_rng(seed)
    nom = read_nominals()
    c = {"corner": np.arange(n),
         "vdd": rng.uniform(0.9 * NOMINAL_VDD, 1.1 * NOMINAL_VDD, n),
         "temp": rng.uniform(-40, 125, n)}
    for dev in ("nmos", "pmos"):
        for name in PROC:
            v = nom[dev][name]
            lo, hi = sorted((0.9 * v, 1.1 * v))          # also right for the negative PMOS vth0
            c[f"{dev[0]}_{name}"] = rng.uniform(lo, hi, n)
    return c


def fmt(v):
    return f"{float(v):.16g}"


# ---- netlists ----------------------------------------------------------------------------------------------------------
def netlist(gate, corner, model_include, combo=None):
    """Transistor-level netlist of one gate at one corner.
    combo given  -> one input combination, one .op (the example netlists);
    combo = None -> all input combinations in one ngspice run (alter + op), used by the runner."""
    g = GATES[gate]
    vdd = corner["vdd"]
    first = combo if combo else input_combos(gate)[0]
    lines = [f"* {gate} transistor-level netlist, leakage at the DC operating point",
             f"* corner {int(corner['corner'])}: VDD = {fmt(vdd)} V, T = {fmt(corner['temp'])} C, inputs {','.join(g['inputs'])} = "
             + (combo_label(combo) if combo else "all combinations (alter + op)"),
             "* all NMOS share the n_* process draw and all PMOS the p_* draw (altermod below); L = 45 nm",
             f'.include "{model_include}"', "", OPTIONS, "", f".temp {fmt(corner['temp'])}", "",
             f"Vdd vdd 0 DC {fmt(vdd)}"]
    for i, name in enumerate(g["inputs"]):
        lines.append(f"V{name} {name} 0 DC {fmt(first[i] * vdd)}")
    lines.append("")
    for (nm, typ, d, gt, s, w) in g["transistors"]:
        body = "vdd" if typ == "pmos" else "0"
        lines.append(f"{nm} {d} {gt} {s} {body} {typ} W={fmt(w)} L={fmt(LMIN)}")
    lines += ["", ".control", "set numdgt=15", ""]
    for dev in ("nmos", "pmos"):
        for name in PROC:
            lines.append(f"altermod {dev} {name} = {fmt(corner[dev[0] + '_' + name])}")
    lines.append("")
    probes = ["i(Vdd)"] + [f"i(V{n})" for n in g["inputs"]] + [f"v({n})" for n in g["nodes"] + [g["out"]]]
    combos = [combo] if combo else input_combos(gate)
    for cb in combos:
        if combo is None:
            for i, name in enumerate(g["inputs"]):
                lines.append(f"alter V{name} = {fmt(cb[i] * vdd)}")
        lines.append("op")
        lines.append(f"echo RESULT {combo_label(cb)} " + " ".join(f"$&{p}" for p in probes))
    lines += ["", ".endc", ".end", ""]
    return "\n".join(lines)


def spice_leak(vdd_src_current, input_src_currents, combo):
    """Leakage in A (positive = drawn from the supplies).
    leak_vdd   = current delivered by the VDD source only
    leak_total = VDD source + every input source that sits at VDD  (= total supply power / VDD)"""
    leak_vdd = -vdd_src_current
    leak_total = leak_vdd - sum(i for i, b in zip(input_src_currents, combo) if b == 1)
    return leak_vdd, leak_total
