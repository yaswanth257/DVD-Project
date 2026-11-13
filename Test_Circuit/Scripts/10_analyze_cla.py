"""Step 10 - Analyse and compare CLA SPICE vs ML predictions.

Merges cla_spice_results.csv and cla_ml_results.csv, computes per-input-combo and
overall error statistics, prints a summary table, writes the results table CSV and the
parity plot (log axes, 128 x 200 = 25,600 points).

Usage (from repository root):
    python3 Test_Circuit/Scripts/10_analyze_cla.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from circuit_common import ML_CSV, PARITY_PNG, SPICE_CSV, TABLE_CSV


def analyse():
    spice = pd.read_csv(SPICE_CSV)
    ml    = pd.read_csv(ML_CSV)

    merged = spice.merge(ml[["corner", "input_combo", "ml_leak_total"]],
                         on=["corner", "input_combo"], how="inner")
    if len(merged) != len(spice):
        raise RuntimeError(f"Row mismatch after merge: {len(merged)} vs {len(spice)}")

    # Error metric (same sign as Part B)
    merged["err_pct"] = (merged["ml_leak_total"] / merged["spice_leak_total"] - 1.0) * 100.0

    def stats(sub):
        return {
            "n_points":          len(sub),
            "spice_median_A":    sub["spice_leak_total"].median(),
            "ml_median_A":       sub["ml_leak_total"].median(),
            "bias_pct":          sub["err_pct"].median(),
            "median_err_pct":    sub["err_pct"].abs().median(),
            "p95_err_pct":       sub["err_pct"].abs().quantile(0.95),
            "mean_accuracy_pct": 100.0 - sub["err_pct"].abs().mean(),
        }

    # Per input combo
    rows = []
    for combo, grp in merged.groupby("input_combo"):
        row = {"input_combo": combo}
        row.update(stats(grp))
        rows.append(row)
    rows.sort(key=lambda r: r["input_combo"])

    # Overall row
    overall = {"input_combo": "ALL"}
    overall.update(stats(merged))
    rows.append(overall)

    table = pd.DataFrame(rows)
    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_CSV, index=False)
    print(f"\nResults table -> {TABLE_CSV}")

    # Print summary
    all_row = table[table["input_combo"] == "ALL"].iloc[0]
    print("\n=== CLA leakage (total): ML vs SPICE ===")
    print(f"  Bias (median signed error) : {all_row['bias_pct']:+.3f} %")
    print(f"  Median absolute error      : {all_row['median_err_pct']:.3f} %")
    print(f"  95th-percentile error      : {all_row['p95_err_pct']:.3f} %")
    print(f"  Mean accuracy              : {all_row['mean_accuracy_pct']:.2f} %")
    print(f"  Total operating points     : {int(all_row['n_points'])}")

    # Print per-combo table header
    print("\nPer input-combo summary (input a2 a1 a0 b2 b1 b0 cin):")
    print(f"{'combo':>7}  {'bias':>7}  {'med err':>7}  {'p95 err':>7}  {'acc':>8}")
    for r in rows:
        if r["input_combo"] == "ALL":
            continue
        print(f"{r['input_combo']:>7}  {r['bias_pct']:>+7.3f}  {r['median_err_pct']:>7.3f}  "
              f"{r['p95_err_pct']:>7.3f}  {r['mean_accuracy_pct']:>8.4f}")

    # Parity plot
    fig, ax = plt.subplots(figsize=(6, 6))
    spice_v = merged["spice_leak_total"].to_numpy()
    ml_v    = merged["ml_leak_total"].to_numpy()
    ax.scatter(spice_v, ml_v, s=1, alpha=0.3, color="steelblue", linewidths=0, rasterized=True)
    lims = [min(spice_v.min(), ml_v.min()) * 0.5, max(spice_v.max(), ml_v.max()) * 2.0]
    ax.plot(lims, lims, color="red", linestyle="dashed", linewidth=1, label="ideal (y = x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("SPICE leakage (A)", fontsize=12)
    ax.set_ylabel("ML leakage (A)", fontsize=12)
    ax.set_title("CLA3 total leakage: ML vs SPICE\n"
                 f"128 combos x 200 corners = {len(merged):,} points\n"
                 f"median err {all_row['median_err_pct']:.3f}%, "
                 f"mean acc {all_row['mean_accuracy_pct']:.2f}%",
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    PARITY_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PARITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nParity plot -> {PARITY_PNG}")


if __name__ == "__main__":
    analyse()
