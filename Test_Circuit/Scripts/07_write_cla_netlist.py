"""Step 7 - Write the CLA3 example netlist (deliverable 2).

Writes a transistor-level SPICE netlist for the 3-bit CLA at corner 0 with all
primary inputs at logic 0. Every gate is expanded to its constituent transistors
(NAND2 sizing Wn/Wp = 180/180 nm, AND2 = NAND2 + INV, OR2 = NOR2 + INV).

Usage (from repository root):
    python3 Test_Circuit/Scripts/07_write_cla_netlist.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Gate_Validation" / "Scripts"))

import pandas as pd
from circuit_common import (CIRCUIT_NAME, CORNERS_CSV, GATES, INPUTS, NETLIST_PATH,
                             LMIN, MODEL_FILE, all_input_combos, combo_label, fmt)
from gate_common import LMIN, MODEL_FILE, OPTIONS, WN_INV, WN_NAND, WP_INV, WP_NAND, WN_NOR, WP_NOR


# ===========================================================================
# Transistor expansion: each gate instance becomes a list of SPICE device lines.
# Net names are prefixed with the instance name to avoid collisions.
# ===========================================================================

def _transistors_for_gate(inst_name, gate_type, in_nets, out_net, vdd_net="vdd"):
    """Return a list of SPICE transistor lines for one gate instance.
    in_nets: list of net names for each gate input (order matches gate_common topology).
    out_net: net name for the gate output.
    Internal nodes get names like <inst_name>_x, <inst_name>_n, etc.
    """
    lines = []
    pfx = inst_name.replace("-", "_")

    def tline(nm, typ, d, g, s, w):
        body = vdd_net if typ == "pmos" else "0"
        return f"M{pfx}_{nm} {d} {g} {s} {body} {typ} W={fmt(w)} L={fmt(LMIN)}"

    if gate_type == "NAND2":
        A, B = in_nets[0], in_nets[1]
        x = f"{pfx}_x"
        lines += [
            tline("mp1", "pmos", out_net, A, vdd_net, WP_NAND),
            tline("mp2", "pmos", out_net, B, vdd_net, WP_NAND),
            tline("mn1", "nmos", out_net, A, x,       WN_NAND),
            tline("mn2", "nmos", x,       B, "0",     WN_NAND),
        ]
    elif gate_type == "AND2":
        # NAND2 driving net n, then INV
        A, B = in_nets[0], in_nets[1]
        x = f"{pfx}_x"
        n = f"{pfx}_n"
        # NAND2 part
        lines += [
            tline("mp1", "pmos", n, A, vdd_net, WP_NAND),
            tline("mp2", "pmos", n, B, vdd_net, WP_NAND),
            tline("mn1", "nmos", n, A, x,       WN_NAND),
            tline("mn2", "nmos", x, B, "0",     WN_NAND),
        ]
        # INV part
        lines += [
            tline("mp3", "pmos", out_net, n, vdd_net, WP_INV),
            tline("mn3", "nmos", out_net, n, "0",     WN_INV),
        ]
    elif gate_type == "OR2":
        # NOR2 driving net n, then INV
        A, B = in_nets[0], in_nets[1]
        y = f"{pfx}_y"
        n = f"{pfx}_n"
        # NOR2 part (series PMOS: y is internal, out=n)
        lines += [
            tline("mp1", "pmos", y,       A, vdd_net, WP_NOR),
            tline("mp2", "pmos", n,       B, y,       WP_NOR),
            tline("mn1", "nmos", n,       A, "0",     WN_NOR),
            tline("mn2", "nmos", n,       B, "0",     WN_NOR),
        ]
        # INV part
        lines += [
            tline("mp3", "pmos", out_net, n, vdd_net, WP_INV),
            tline("mn3", "nmos", out_net, n, "0",     WN_INV),
        ]
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")
    return lines


def build_netlist(corner, combo=None):
    """Build a full transistor-level SPICE netlist for the CLA at one corner.
    If combo is None the first input combination (all zeros) is used for the initial
    DC values of the input sources, which forms the example netlist.
    combo: tuple of 7 ints (a0, a1, a2, b0, b1, b2, cin) or None.
    """
    if combo is None:
        combo = (0,) * 7
    vdd = corner["vdd"]

    lines = [
        f"* {CIRCUIT_NAME} 3-bit Carry-Lookahead Adder transistor-level netlist",
        f"* corner {int(corner['corner'])}: VDD = {fmt(vdd)} V, T = {fmt(corner['temp'])} C",
        f"* inputs a2 a1 a0 b2 b1 b0 cin = {combo_label(combo)}",
        "* gate sizing from gate_common: NAND2 Wn/Wp 180/180 nm, AND2 (NAND+INV), OR2 (NOR+INV)",
        f'.include "{MODEL_FILE}"',
        "",
        OPTIONS,
        "",
        f".temp {fmt(corner['temp'])}",
        "",
        f"Vdd vdd 0 DC {fmt(vdd)}",
    ]
    for i, name in enumerate(INPUTS):
        lines.append(f"V{name} {name} 0 DC {fmt(combo[i] * vdd)}")
    lines.append("")

    # Expand every gate to transistors
    for (inst, gate_type, in_nets, out_net) in GATES:
        lines.append(f"* {inst}: {gate_type}({', '.join(in_nets)}) -> {out_net}")
        lines.extend(_transistors_for_gate(inst, gate_type, in_nets, out_net))
        lines.append("")

    # Simple .op analysis block
    lines += [".control", "set numdgt=15", ""]
    for dev in ("nmos", "pmos"):
        from gate_common import PROC as _PROC
        for pname in _PROC:
            lines.append(f"altermod {dev} {pname} = {fmt(corner[dev[0] + '_' + pname])}")
    lines += ["", "op", ".endc", ".end", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    corners = pd.read_csv(CORNERS_CSV).to_dict("records")
    corner0 = corners[0]
    combo0  = (0, 0, 0, 0, 0, 0, 0)
    text    = build_netlist(corner0, combo0)
    NETLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    NETLIST_PATH.write_text(text)
    print(f"Wrote {NETLIST_PATH}")
    print(f"Corner 0: VDD = {corner0['vdd']:.4f} V, T = {corner0['temp']:.2f} C, inputs = all 0")
    print(f"Total lines: {len(text.splitlines())}")
