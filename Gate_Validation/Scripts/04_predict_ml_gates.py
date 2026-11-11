"""Step 4 - ML prediction of gate leakage from the four transistor models only (no SPICE currents are used).

For every gate / input combination / corner of step 3, the SPICE internal-node and output voltages give the terminal voltages
(vd, vg, vs, vb) of every transistor. Each transistor row (plus W and the corner's VDD, T and NMOS / PMOS process values)
goes to the matching model (NMOS_ON / NMOS_OFF / PMOS_ON / PMOS_OFF, chosen by the gate voltage), which returns id, ig, is, ib.
Leakage is then summed from the predicted terminal currents (sign: + = out of the transistor):
    ml_leak_vdd   = - sum of currents at terminals tied to the VDD net
    ml_leak_total = ml_leak_vdd - sum of currents at terminals tied to an input net that is at logic 1   (matches SPICE leak_total)
ml_kcl_<node> = sum of predicted currents leaving the transistors at that node (should be ~0; diagnostic for a later node solver).

    python3 Gate_Validation/Scripts/04_predict_ml_gates.py
"""
import numpy as np
import pandas as pd
from gate_common import (CORNERS_CSV, GATES, GATE_ORDER, ML_CSV, ROOT, SPICE_CSV, LMIN)
from numpy_model import load_model

models = {c: load_model(ROOT / "ML_Models" / "Saved_Models" / f"{c}.pt") for c in ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]}
print("model backend:", type(next(iter(models.values()))).__name__)
corners = pd.read_csv(CORNERS_CSV).set_index("corner")
spice = pd.read_csv(SPICE_CSV, dtype={"input": str})
FEAT = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
out_frames = []

for gate in GATE_ORDER:
    g = GATES[gate]
    s = spice[spice.gate == gate].reset_index(drop=True)
    vdd = s["vdd"].to_numpy()
    c = corners.loc[s["corner"]].reset_index(drop=True)
    bits = {name: np.array([int(lab[i]) for lab in s["input"]]) for i, name in enumerate(g["inputs"])}
    volt = {"vdd": vdd, "0": np.zeros(len(s))}
    for name in g["inputs"]:
        volt[name] = bits[name] * vdd                                   # inputs are exactly 0 or VDD
    for node in g["nodes"] + [g["out"]]:
        volt[node] = s[f"spice_v_{node}"].to_numpy()
    kcl = {net: np.zeros(len(s)) for net in g["nodes"] + [g["out"]]}
    leak_vdd = np.zeros(len(s)); leak_in = np.zeros(len(s))

    for (name, typ, d, gt, src, w) in g["transistors"]:
        dev = typ[0]                                                    # 'n' / 'p' -> process columns n_* / p_*
        body_net = "vdd" if typ == "pmos" else "0"
        clip = lambda v: np.clip(v, 0.0, vdd)                           # SPICE nodes can sit a few uV outside [0, VDD]
        row = pd.DataFrame({"vdd": vdd, "temp": s["temp"], "w": w, "vd": clip(volt[d]), "vs": clip(volt[src]),
                            "vg": clip(volt[gt]), "vb": volt[body_net],
                            **{p: c[f"{dev}_{p}"] for p in ["vth0", "toxe", "ndep", "u0", "xj"]}})[FEAT]
        is_on = (row["vg"] > vdd / 2) if typ == "nmos" else (row["vg"] < vdd / 2)
        cur = pd.DataFrame(0.0, index=row.index, columns=["id", "ig", "is", "ib"])
        for state, mask in (("ON", is_on), ("OFF", ~is_on)):
            if mask.any():
                cur.loc[mask] = models[f"{typ.upper()}_{state}"].predict_signed(row[mask]).to_numpy()
        for term, net in (("id", d), ("ig", gt), ("is", src), ("ib", body_net)):
            i_out = cur[term].to_numpy()
            if net == "vdd":
                leak_vdd -= i_out
            elif net in g["inputs"]:
                leak_in -= i_out * bits[net]                            # only inputs at logic 1 deliver power
            elif net in kcl:
                kcl[net] += i_out

    o = s[["gate", "input", "corner", "vdd", "temp"]].copy()
    o["ml_leak_vdd"], o["ml_leak_total"] = leak_vdd, leak_vdd + leak_in
    for node in g["nodes"] + [g["out"]]:
        o[f"ml_v_{node}"] = volt[node]                                  # node voltages used (taken from SPICE, step 4 of the brief)
        o[f"ml_kcl_{node}"] = kcl[node]
    out_frames.append(o)

ml = pd.concat(out_frames, ignore_index=True)
ml.to_csv(ML_CSV, index=False)
print(f"{len(ml)} ML predictions -> {ML_CSV}")
