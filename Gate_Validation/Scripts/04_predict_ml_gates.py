"""Step 4 - ML prediction of gate leakage from the four transistor models only (no SPICE currents are used).

For every gate / input combination / corner of step 3, the SPICE internal-node and output voltages give the terminal voltages
(vd, vg, vs, vb) of every transistor. Each transistor row (plus W and the corner's VDD, T and NMOS / PMOS process values)
goes to the matching model (NMOS_ON / NMOS_OFF / PMOS_ON / PMOS_OFF, chosen by the gate voltage), which returns id, ig, is, ib
with their sign (+ = current out of the transistor).

Composition (see gate_common.compose_currents): the channel current of every ON transistor is taken from Kirchhoff's current
law at the internal nodes, not from the model. An ON transistor in a gate sits at |Vds| of 20 uV - 1 mV and carries the
leakage of the OFF transistors through its channel; that current is fixed by the rest of the gate, and the model cannot
resolve it from such a small Vds (median error 8 - 93 % for its id / is; its ig and ib stay with the model). Leakage is then
    ml_leak_vdd   = - sum of currents at terminals tied to the VDD net
    ml_leak_total = ml_leak_vdd - sum of currents at terminals tied to an input net that is at logic 1   (matches SPICE leak_total)
ml_leak_*_sum is the plain sum of the model currents (no Kirchhoff step), kept for comparison; ml_kcl_<node> is the sum of the
plain model currents leaving the transistors at that node, i.e. how far they are from Kirchhoff (0 after the composition).

    python3 Gate_Validation/Scripts/04_predict_ml_gates.py
"""
import numpy as np
import pandas as pd
from gate_common import CORNERS_CSV, GATES, GATE_ORDER, ML_CSV, ROOT, SPICE_CSV, compose_currents, input_combos, combo_label
from numpy_model import load_model

models = {c: load_model(ROOT / "ML_Models" / "Saved_Models" / f"{c}.pt") for c in ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]}
print("model backend:", type(next(iter(models.values()))).__name__)
corners = pd.read_csv(CORNERS_CSV).set_index("corner")
spice = pd.read_csv(SPICE_CSV, dtype={"input": str})
FEAT = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
out_frames = []

for gate in GATE_ORDER:
    g = GATES[gate]
    for combo in input_combos(gate):
        s = spice[(spice.gate == gate) & (spice.input == combo_label(combo))].reset_index(drop=True)
        n, vdd = len(s), s["vdd"].to_numpy()
        c = corners.loc[s["corner"]].reset_index(drop=True)
        bits = dict(zip(g["inputs"], combo))
        volt = {"vdd": vdd, "0": np.zeros(n), **{name: b * vdd for name, b in bits.items()}}   # inputs are exactly 0 or VDD
        for node in g["nodes"] + [g["out"]]:
            volt[node] = s[f"spice_v_{node}"].to_numpy()

        currents, on = {}, {}
        for (name, typ, d, gt, src, w) in g["transistors"]:
            body_net = "vdd" if typ == "pmos" else "0"
            clip = lambda v: np.clip(v, 0.0, vdd)                       # SPICE nodes can sit a few uV outside [0, VDD]
            row = pd.DataFrame({"vdd": vdd, "temp": s["temp"], "w": w, "vd": clip(volt[d]), "vs": clip(volt[src]),
                                "vg": clip(volt[gt]), "vb": volt[body_net],
                                **{p: c[f"{typ[0]}_{p}"] for p in ["vth0", "toxe", "ndep", "u0", "xj"]}})[FEAT]
            vg = row["vg"].to_numpy()                                   # ON / OFF from the gate voltage (inputs or node n)
            state = (vg > vdd / 2) if typ == "nmos" else (vg < vdd / 2)
            assert state.all() or not state.any(), f"{gate} {combo}: {name} changes state between corners"
            on[name] = bool(state[0])
            currents[name] = models[f"{typ.upper()}_{'ON' if on[name] else 'OFF'}"].predict_signed(row).to_numpy()

        o = s[["gate", "input", "corner", "vdd", "temp"]].copy()
        for mode, kcl in (("", True), ("_sum", False)):
            leak_vdd, leak_total, residual = compose_currents(gate, combo, currents, on, kcl=kcl)
            o[f"ml_leak_vdd{mode}"], o[f"ml_leak_total{mode}"] = leak_vdd, leak_total
            if not kcl:
                for node, r in residual.items():
                    o[f"ml_kcl_{node}"] = r
        for node in g["nodes"] + [g["out"]]:
            o[f"ml_v_{node}"] = volt[node]                              # node voltages used (taken from SPICE, step 4 of the brief)
        out_frames.append(o)

ml = pd.concat(out_frames, ignore_index=True)
ml.to_csv(ML_CSV, index=False)
print(f"{len(ml)} ML predictions -> {ML_CSV}")
