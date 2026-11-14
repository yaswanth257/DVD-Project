"""Step 5 - compare ML with SPICE for every gate and input combination (deliverables 6 and 7).

Per-case metrics (200 corners each):   error % = (ML / SPICE - 1) x 100
  bias = median signed error, median err / 95th pct err = percentiles of |error|, mean accuracy = 100 - mean |error|.
Two leakage definitions are reported (see 04_predict_ml_gates.py): 'total' (VDD source + inputs at VDD) and 'vdd' (VDD source only).
The tables use the Kirchhoff composition of step 4; gate_composition_comparison.csv puts it next to the plain sum of the model currents.

    python3 Gate_Validation/Scripts/05_analyze_gates.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gate_common import GV, ML_CSV, SPICE_CSV, GATE_ORDER, case_list

s = pd.read_csv(SPICE_CSV, dtype={"input": str})
m = pd.read_csv(ML_CSV, dtype={"input": str})
# join on the identifiers only (vdd / temp are floats; a last-digit difference must not drop rows)
d = s.merge(m.drop(columns=["vdd", "temp"]), on=["gate", "input", "corner"])
if len(d) != len(s):
    raise RuntimeError(f"SPICE and ML results do not match: {len(d)} of {len(s)} operating points after the merge")
d.to_csv(GV / "Results" / "gate_spice_vs_ml.csv", index=False)

for kind in ("total", "vdd"):
    d[f"err_{kind}"] = (d[f"ml_leak_{kind}"] / d[f"spice_leak_{kind}"] - 1) * 100
    rows = []
    for gate, inp in case_list():
        x = d[(d.gate == gate) & (d.input == inp)]
        e = x[f"err_{kind}"]
        rows.append({"gate": gate, "input": inp, "SPICE median (A)": x[f"spice_leak_{kind}"].median(),
                     "ML median (A)": x[f"ml_leak_{kind}"].median(), "bias %": e.median(),
                     "median err %": e.abs().median(), "95th pct err %": e.abs().quantile(0.95),
                     "mean accuracy %": 100 - e.abs().mean()})
    t = pd.DataFrame(rows)
    allrow = {"gate": "ALL", "input": "-", "SPICE median (A)": d[f"spice_leak_{kind}"].median(),
              "ML median (A)": d[f"ml_leak_{kind}"].median(), "bias %": d[f"err_{kind}"].median(),
              "median err %": d[f"err_{kind}"].abs().median(), "95th pct err %": d[f"err_{kind}"].abs().quantile(0.95),
              "mean accuracy %": 100 - d[f"err_{kind}"].abs().mean()}
    t = pd.concat([t, pd.DataFrame([allrow])], ignore_index=True)
    t.to_csv(GV / "Results" / f"gate_results_table_{kind}.csv", index=False)
    print(f"\n=== Gate results, leakage = {kind} ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig, ax = plt.subplots(figsize=(6, 6))
    for gate, mk in zip(GATE_ORDER, "osv^D"):
        x = d[d.gate == gate]
        ax.scatter(x[f"spice_leak_{kind}"], x[f"ml_leak_{kind}"], s=8, marker=mk, alpha=0.5, label=gate)
    lo, hi = d[f"spice_leak_{kind}"].min() * 0.8, d[f"spice_leak_{kind}"].max() * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_aspect("equal")
    ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)")
    ax.set_title(f"Gate leakage, ML vs SPICE ({len(d)} simulations, {kind})"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(GV / "Plots" / f"gate_parity_{kind}.png", dpi=150); plt.close(fig)

# effect of the composition: Kirchhoff for the ON channels (used above) vs the plain sum of the model currents
rows = []
for kind in ("total", "vdd"):
    for label, col in (("Kirchhoff for ON channels", f"ml_leak_{kind}"), ("plain sum of model currents", f"ml_leak_{kind}_sum")):
        e = (d[col] / d[f"spice_leak_{kind}"] - 1) * 100
        rows.append({"leakage": kind, "composition": label, "bias %": e.median(), "median err %": e.abs().median(),
                     "95th pct err %": e.abs().quantile(0.95), "mean accuracy %": 100 - e.abs().mean()})
comp = pd.DataFrame(rows)
comp.to_csv(GV / "Results" / "gate_composition_comparison.csv", index=False)
print("\n=== Composition: Kirchhoff for ON channels vs plain sum (all 3,600 operating points) ===")
print(comp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# error vs temperature / VDD, per gate (for the error analysis in the report)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for gate in GATE_ORDER:
    x = d[d.gate == gate]
    axs[0].scatter(x.temp, x.err_total, s=4, alpha=0.4, label=gate)
    axs[1].scatter(x.vdd, x.err_total, s=4, alpha=0.4, label=gate)
axs[0].set_xlabel("temperature (C)"); axs[1].set_xlabel("VDD (V)")
for a in axs: a.set_ylabel("ML error vs SPICE (%)"); a.axhline(0, color="k", lw=0.8); a.grid(alpha=0.3)
axs[0].legend(markerscale=3); fig.tight_layout(); fig.savefig(GV / "Plots" / "gate_error_vs_conditions.png", dpi=150)
