"""Install the final models: the Huber-loss MLP of each case, chosen on the validation set, saved as Saved_Models/<case>.pt.

    python3 ML_Models/Scripts/export_huber_models.py        (after run_offstudy.py --stage 13)

run_offstudy.py --stage 13 trains two seeds per case (MLP 128-64-32-16, Huber loss with the transition width chosen on
validation, β = 0.002) and leaves the candidates in Saved_Models/huber/<case>__seed*.pt. For each case the candidate with the
lower validation median error is kept; a candidate whose worst validation prediction is off by more than MAX_DLOG decades (a
numerical blow-up of the network, seen once in the study) is not eligible. The chosen model is written to Saved_Models/<case>.pt
with the metadata that Results/model_evaluation.ipynb reads (model name, the validation errors recorded when it was trained, the
loss), and the candidates are deleted.

The script also writes Results/validation_comparison_HUBER.csv: the validation errors of the four final models in the format of
the comparison tables of the training notebooks, so that model_evaluation.ipynb lists the Huber MLP next to the other model types.
The test set is not used here; model_evaluation.ipynb evaluates the final models on it once.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from leakage_model import SEED, LeakageModel, OUTPUT_COLS, error_table, split_indices  # noqa: E402

ROOT = HERE.parents[1]
MODELS = ROOT / "ML_Models" / "Saved_Models"
CANDIDATES = MODELS / "huber"
RUNS = ROOT / "ML_Models" / "Results" / "offstudy" / "final" / "runs"
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
MODEL_NAME = "MLP (Huber loss)"
METRICS = ["R2 (log|I|)", "mean err %", "median err %", "95th pct err %"]
MAX_DLOG = 6.0


def training_run(case, seed):
    """The run record (metrics at training time) that run_offstudy.py --stage 13 wrote for this case and seed."""
    found = sorted(RUNS.glob(f"{case}__random__log__lam0__seed{seed}__pat3000__huber*.json"))
    assert len(found) == 1, f"{case} seed {seed}: expected one run record in {RUNS}, found {len(found)}"
    return json.loads(found[0].read_text())


rows = []
for case in CASES:
    df = pd.read_csv(ROOT / "Data_Generation" / "Datasets" / f"{case}_20k.csv")
    _, idx_val, _ = split_indices(len(df))
    y_val = np.log10(df.iloc[idx_val][OUTPUT_COLS].abs().to_numpy())
    # candidates of stage 13 (or a model an earlier version of this script left as Saved_Models/huber/<case>.pt)
    candidates = sorted(CANDIDATES.glob(f"{case}__seed*.pt")) or sorted(CANDIDATES.glob(f"{case}.pt"))
    if candidates:
        scored = []
        for f in candidates:
            pred = LeakageModel(f).predict_log(df.iloc[idx_val])
            worst = float(np.max(np.abs(pred - y_val)))
            median = error_table(y_val, pred).loc["average", "median err %"]
            scored.append((median if worst < MAX_DLOG else np.inf, worst, f))
        scored.sort(key=lambda s: s[0])
        assert np.isfinite(scored[0][0]), f"{case}: every candidate has a blown-up prediction"
        best = scored[0][2]
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        run = training_run(case, ckpt["seed"])
        # "seed" is the training seed of the network; the 80/10/10 split always uses SEED (split_seed)
        ckpt.update(model=MODEL_NAME, loss={"kind": "huber", "beta": run["huber_beta"]}, epochs=run["epochs"],
                    split_seed=SEED, training_run=run["tag"], validation={k: run["val"]["average"][k] for k in METRICS})
        torch.save(ckpt, MODELS / f"{case}.pt")
        print(f"{case}: candidates " + ", ".join(f"seed {torch.load(f, weights_only=True)['seed']} {v:.3f} % (worst {w:.1f} decades)"
                                                for v, w, f in scored) + f" -> seed {ckpt['seed']} saved as {(MODELS / f'{case}.pt').relative_to(ROOT)}")
        for f in candidates:
            f.unlink()
    ckpt = torch.load(MODELS / f"{case}.pt", map_location="cpu", weights_only=True)
    assert ckpt.get("model") == MODEL_NAME, f"{case}: Saved_Models/{case}.pt is not a Huber model; run run_offstudy.py --stage 13 first"
    run = training_run(case, ckpt["seed"])
    for current in OUTPUT_COLS + ["average"]:
        rows.append({"case": case, "model": MODEL_NAME, "current": current,
                     **{k: run["val"][current][k] for k in METRICS}, "train time s": run["seconds"]})

if CANDIDATES.is_dir() and not any(CANDIDATES.iterdir()):
    CANDIDATES.rmdir()
out = ROOT / "ML_Models" / "Results" / "validation_comparison_HUBER.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("written", out.relative_to(ROOT))
