"""Check the current-sign rule of leakage_model.SIGN_RULES (used by LeakageModel.predict_signed for Parts B and C).

    python3 ML_Models/Scripts/check_sign_rule.py        (from the repository root; writes ML_Models/Results/sign_rule_check.csv)

1. The three fixed signs of every case (ig, ib, and the fixed drain / source terminal) on all 20,000 rows of each dataset.
2. The full rule (fixed signs + Kirchhoff for the remaining terminal) with the TRUE magnitudes, all rows.
3. The full rule with the PREDICTED magnitudes of the final models (Saved_Models/<case>.pt), test set (what Parts B / C will see).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from leakage_model import OUTPUT_COLS, SIGN_RULES, LeakageModel, apply_signs, split_indices  # noqa: E402

rows = []
for case, (s_g, s_b, which, s_fix) in SIGN_RULES.items():
    df = pd.read_csv(ROOT / "Data_Generation" / "Datasets" / f"{case}_20k.csv")
    I = df[OUTPUT_COLS].to_numpy()
    fix_is_d = (df.vd >= df.vs).to_numpy() if which == "high" else (df.vd < df.vs).to_numpy()
    fixed = {"ig": np.mean(np.sign(I[:, 1]) == s_g), "ib": np.mean(np.sign(I[:, 3]) == s_b),
             f"{which}-voltage terminal": np.mean(np.sign(np.where(fix_is_d, I[:, 0], I[:, 2])) == s_fix)}
    print(f"{case:8s} fixed signs on all {len(df)} rows: " + ", ".join(f"{k} {'+' if s > 0 else '-'} {v:.2%}"
          for (k, v), s in zip(fixed.items(), (s_g, s_b, s_fix))))
    test = split_indices(len(df))[2]
    for source, idx, mags in (("true magnitudes", np.arange(len(df)), np.abs(I)), ("final model", test, None)):
        if mags is None:
            P = LeakageModel(ROOT / "ML_Models" / "Saved_Models" / f"{case}.pt").predict_signed(df.iloc[idx]).to_numpy()
        else:
            P = apply_signs(case, mags, df.vd.to_numpy(), df.vs.to_numpy())
        T = I[idx]
        wrong = np.sign(P) != np.sign(T)
        rows.append({"case": case, "magnitudes": source, "rows": len(idx),
                     "all four signs right %": 100 * np.mean(~wrong.any(1)),
                     "wrongly signed |I| / total |I|": np.abs(T)[wrong].sum() / np.abs(T).sum(),
                     "largest wrongly signed current / its row max": float(((np.abs(T) * wrong).max(1) / np.abs(T).max(1)).max())})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
out = ROOT / "ML_Models" / "Results" / "sign_rule_check.csv"
res.to_csv(out, index=False)
print("written", out.relative_to(ROOT))
