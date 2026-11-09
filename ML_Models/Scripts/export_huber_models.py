"""Pick the final Huber-loss model of each case, evaluate it on the test set next to the official model, plot it.

    python3 ML_Models/Scripts/export_huber_models.py        (after run_offstudy.py --stage 13)

For every case the two trained seeds (Saved_Models/huber/<case>__seed*.pt) are compared on the VALIDATION set; the one with the
lower validation median error is kept as Saved_Models/huber/<case>.pt and the other weights are deleted. A candidate whose worst
validation prediction is off by more than MAX_DLOG decades (a numerical blow-up of the network, seen once in the study) is not eligible. The test set is used
once, for the comparison with the official model (Saved_Models/<case>.pt) in Results/huber_test_results.csv and the parity plots.
The official models are not modified. The Huber models load with `LeakageModel` like the official ones.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offstudy as o                                                      # noqa: E402
from leakage_model import LeakageModel, error_table, split_indices, OUTPUT_COLS   # noqa: E402

ROOT = o.ROOT
MODELS, HUBER = ROOT / "ML_Models" / "Saved_Models", ROOT / "ML_Models" / "Saved_Models" / "huber"
FINAL_RUNS = ROOT / "ML_Models" / "Results" / "offstudy" / "final" / "runs"
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
MAX_DLOG = 6.0
log_true = lambda d: np.log10(d[OUTPUT_COLS].abs().to_numpy())

rows, preds = [], {}
for case in CASES:
    df = pd.read_csv(o.DATA_DIR / f"{case}_20k.csv")
    _, iv, it = split_indices(len(df))
    cand = sorted(HUBER.glob(f"{case}__seed*.pt"))
    if cand:                                              # choose on the validation set
        scored = []
        for f in cand:
            m = LeakageModel(f)
            pv = m.predict_log(df.iloc[iv])
            val = error_table(log_true(df.iloc[iv]), pv).loc["average"]
            dlog = float(np.max(np.abs(pv - log_true(df.iloc[iv]))))
            scored.append((val["median err %"] if dlog < MAX_DLOG else np.inf, dlog, f))
        scored.sort(key=lambda x: x[0])
        best = scored[0][2]
        assert np.isfinite(scored[0][0]), f"{case}: every candidate has a blown-up prediction"
        print(f"{case}: validation median (worst |dlog10|) " + ", ".join(f"{f.stem.split('__')[1]} {v:.3f} % ({dl:.1f})" for v, dl, f in scored) + f" -> {best.stem.split('__')[1]}")
        best.replace(HUBER / f"{case}.pt")
        for f in cand:
            if f.exists():
                f.unlink()
    final = HUBER / f"{case}.pt"
    for name, path in (("official (notebooks)", MODELS / f"{case}.pt"), ("Huber loss", final)):
        m = LeakageModel(path)
        for part, idx in (("validation", iv), ("test", it)):
            d = df.iloc[idx]
            plog = m.predict_log(d)
            tab = error_table(log_true(d), plog)
            pred_mag = m.predict(d)
            err = (pred_mag - d[OUTPUT_COLS].abs()).abs()
            a = tab.loc["average"]
            rows.append({"case": case, "model": name, "set": part, "R2 (log|I|)": a["R2 (log|I|)"], "median err %": a["median err %"],
                         "95th pct err %": a["95th pct err %"], "mean err %": a["mean err %"],
                         "id median / 95th / mean %": " / ".join(f"{tab.loc['id', k]:.2f}" for k in ("median err %", "95th pct err %", "mean err %")),
                         "is median / 95th / mean %": " / ".join(f"{tab.loc['is', k]:.2f}" for k in ("median err %", "95th pct err %", "mean err %")),
                         "sum err % (A-weighted)": 100 * err.to_numpy().sum() / d[OUTPUT_COLS].abs().to_numpy().sum(),
                         "mean abs err id (A)": err["id"].mean(), "worst |log10 error| (decades)": float(np.max(np.abs(plog - log_true(d))))})
        if name == "Huber loss":
            preds[case] = (d[OUTPUT_COLS].abs(), pred_mag)         # test rows

res = pd.DataFrame(rows)
res.to_csv(ROOT / "ML_Models" / "Results" / "huber_test_results.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
print(res[res["set"] == "test"].drop(columns="set").to_string(index=False))

# parity plots: 4 currents x 4 cases, test set, log axes
BLUE, INK_2, GRID, SURFACE = "#2a78d6", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                     "ytick.color": INK_2, "font.size": 8, "axes.titlesize": 9})
fig, axes = plt.subplots(4, 4, figsize=(12, 11.5))
for j, case in enumerate(CASES):
    true, pred = preds[case]
    for i, c in enumerate(OUTPUT_COLS):
        ax = axes[i, j]
        ax.scatter(true[c], pred[c], s=2, alpha=0.35, color=BLUE, linewidths=0, rasterized=True)
        lo, hi = min(true[c].min(), pred[c].min()), max(true[c].max(), pred[c].max())
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.8)
        ax.set(xscale="log", yscale="log")
        if i == 0:
            ax.set_title(case)
        if j == 0:
            ax.set_ylabel(f"ML |{c}| (A)")
        if i == 3:
            ax.set_xlabel("SPICE |I| (A)")
fig.suptitle("Huber-loss models, test set (2,000 rows per case): predicted vs SPICE", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "ML_Models" / "Plots" / "parity_huber_models.png", dpi=130)
print("saved", HUBER, ROOT / "ML_Models" / "Plots" / "parity_huber_models.png")
