"""Step 6 (diagnostic) - why is the gate-level error larger than the transistor-level error?

For a subset of corners, every transistor of every gate is re-simulated as a standalone single-FET netlist (same recipe as the
training data) at exactly the terminal voltages it sees inside the gate. Two things follow:
  (a) summing those single-FET SPICE currents with the leakage rule must reproduce the gate SPICE leakage -> the composition
      (terminal-to-net mapping, signs, leakage definition) is correct;
  (b) ML vs single-FET SPICE, per transistor and per current, binned by |Vds| and ON/OFF state -> where the models fail.

    python3 Gate_Validation/Scripts/06_diagnose_models_in_gates.py [n_corners]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from gate_common import CORNERS_CSV, GATES, GATE_ORDER, GV, MODEL_FILE, NGSPICE, OPTIONS, PROC, ROOT, SPICE_CSV, fmt
from numpy_model import load_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
models = {c: load_model(ROOT / "ML_Models" / "Saved_Models" / f"{c}.pt") for c in ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]}
corners = pd.read_csv(CORNERS_CSV).iloc[:N]
sp = pd.read_csv(SPICE_CSV, dtype={"input": str})


def fet_spice(typ, w, vd, vg, vs, vb, cor, tmp):
    alt = "\n".join(f"altermod {typ} {p} = {fmt(cor[typ[0] + '_' + p])}" for p in PROC)
    path = Path(tmp) / "f.cir"
    path.write_text(f""".include "{MODEL_FILE}"
{OPTIONS}
.temp {fmt(cor['temp'])}
M1 drain gate source body {typ} W={fmt(w)} L=45n
Vd drain 0 DC {fmt(vd)}
Vg gate 0 DC {fmt(vg)}
Vs source 0 DC {fmt(vs)}
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
    return np.array([float(x) for x in [l for l in o.splitlines() if l.startswith("RESULT")][0].split()[1:]])


rec, comp = [], []
with tempfile.TemporaryDirectory() as tmp:
    for gate in GATE_ORDER:
        g = GATES[gate]
        for k, cor in corners.iterrows():
            vdd = cor["vdd"]
            for lab in sorted(set(sp[sp.gate == gate].input)):
                r = sp[(sp.gate == gate) & (sp.input == lab) & (sp.corner == k)].iloc[0]
                bits = {n: int(lab[i]) for i, n in enumerate(g["inputs"])}
                v = {"vdd": vdd, "0": 0.0, **{n: b * vdd for n, b in bits.items()}}
                for n in g["nodes"] + [g["out"]]:
                    v[n] = r[f"spice_v_{n}"]
                leak_f = leak_m = 0.0
                for (nm, typ, d, gt, s, w) in g["transistors"]:
                    vb = vdd if typ == "pmos" else 0.0
                    vg = v[gt]
                    on = (vg > vdd / 2) if typ == "nmos" else (vg < vdd / 2)
                    f = fet_spice(typ, w, v[d], vg, v[s], vb, cor, tmp)
                    row = pd.DataFrame({"vdd": [vdd], "temp": [cor["temp"]], "w": [w], "vd": [np.clip(v[d], 0, vdd)],
                                        "vs": [np.clip(v[s], 0, vdd)], "vg": [np.clip(vg, 0, vdd)], "vb": [vb],
                                        **{p: [cor[typ[0] + "_" + p]] for p in PROC}})
                    mcur = models[f"{typ.upper()}_{'ON' if on else 'OFF'}"].predict_signed(row).to_numpy()[0]
                    for term, net, fi, mi in zip(("id", "ig", "is", "ib"), (d, gt, s, vb and "vdd" or "0"), f, mcur):
                        net = "vdd" if (term == "ib" and typ == "pmos") else ("0" if term == "ib" else net)
                        if net == "vdd" or (net in bits and bits[net] == 1):
                            leak_f -= fi; leak_m -= mi
                    for term, fi, mi in zip(("id", "ig", "is", "ib"), f, mcur):
                        rec.append({"gate": gate, "input": lab, "corner": k, "fet": nm, "type": typ, "state": "ON" if on else "OFF",
                                    "current": term, "vds_abs": abs(v[d] - v[s]), "spice": fi, "ml": mi})
                comp.append({"gate": gate, "input": lab, "corner": k, "gate_spice": r.spice_leak_total, "fet_sum": leak_f, "ml": leak_m})

c = pd.DataFrame(comp)
c["composition_err_%"] = (c.fet_sum / c.gate_spice - 1) * 100
print(f"(a) single-FET SPICE currents summed vs gate SPICE ({len(c)} cases): median |err| {c['composition_err_%'].abs().median():.4f} %, "
      f"max {c['composition_err_%'].abs().max():.3f} %")
r = pd.DataFrame(rec)
r["err_%"] = (r.ml / r.spice - 1) * 100
r["abs_err_%"] = r["err_%"].abs()
r["vds bin"] = pd.cut(r.vds_abs, [-1, 1e-3, 1e-2, 0.1, 0.5, 2], labels=["<1 mV", "1-10 mV", "10-100 mV", "0.1-0.5 V", ">0.5 V"])
r.to_csv(GV / "Results" / "diag_per_transistor.csv", index=False)
big = r[r.spice.abs() > 1e-10]                       # skip currents that are numerically ~0
t = big.groupby(["type", "state", "vds bin"], observed=True)["abs_err_%"].agg(["median", lambda x: x.quantile(0.95), "count"])
t.columns = ["median |err| %", "95th pct |err| %", "count"]
print("\n(b) ML vs single-FET SPICE inside gates, by transistor state and |Vds| (currents > 0.1 nA):")
print(t.round(2).to_string())
t.to_csv(GV / "Results" / "diag_error_by_vds.csv")
