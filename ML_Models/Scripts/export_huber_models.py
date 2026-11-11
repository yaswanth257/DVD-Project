"""Install the final OFF models (the Huber-loss MLPs of run_offstudy.py --stage 13) and write the Huber comparison rows.

    python3 ML_Models/Scripts/export_huber_models.py        (after run_offstudy.py --stage 13; then train_on_models_gate_region.py)

run_offstudy.py --stage 13 trains two seeds per case (MLP 128-64-32-16, Huber loss, β = 0.002 chosen on validation), records
each run in Results/offstudy/final/{runs,preds}/ and leaves the weights in Saved_Models/huber/<case>__seed*.pt. Per case the seed
with the lower validation median error is chosen; a run whose worst validation prediction is off by more than MAX_DLOG decades
(a numerical blow-up of the network, seen once in the study) is not eligible.
  * NMOS_OFF, PMOS_OFF: the chosen model is installed as Saved_Models/<case>.pt with the metadata model_evaluation.ipynb reads
    (model name, validation errors recorded at training time, loss, training seed and split seed).
  * NMOS_ON, PMOS_ON: their final models are trained with extra gate-region rows by train_on_models_gate_region.py; the stage-13
    Huber models only provide their comparison rows.
Results/validation_comparison_HUBER.csv gets the validation errors of the chosen stage-13 model of all four cases, in the
format of the training notebooks' tables (model_evaluation.ipynb lists it as a model variant). The candidate weights are deleted.
Without candidates (already installed) the script only rewrites the comparison rows and checks the installed OFF models.
The test set is not used here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from leakage_model import SEED  # noqa: E402

ROOT = HERE.parents[1]
MODELS = ROOT / "ML_Models" / "Saved_Models"
CANDIDATES = MODELS / "huber"
FINAL = ROOT / "ML_Models" / "Results" / "offstudy" / "final"
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
INSTALL = ["NMOS_OFF", "PMOS_OFF"]
MODEL_NAME = "MLP (Huber loss)"
METRICS = ["R2 (log|I|)", "mean err %", "median err %", "95th pct err %"]
MAX_DLOG = 6.0

rows = []
for case in CASES:
    scored = []
    for p in sorted((FINAL / "runs").glob(f"{case}__random__log__lam0__seed*__pat3000__huber*.json")):
        run = json.loads(p.read_text())
        z = np.load(FINAL / "preds" / f"{run['tag']}.npz")
        worst = float(np.max(np.abs(np.log10(np.abs(z["val"])) - np.log10(np.abs(z["val_true"])))))
        scored.append((run["val"]["average"]["median err %"] if worst < MAX_DLOG else np.inf, worst, run))
    assert scored, f"{case}: no stage-13 run records in {FINAL / 'runs'}"
    scored.sort(key=lambda s: s[0])
    assert np.isfinite(scored[0][0]), f"{case}: every run has a blown-up prediction"
    run = scored[0][2]
    print(f"{case}: " + ", ".join(f"seed {r['seed']} {v:.3f} % (worst {w:.1f} decades)" for v, w, r in scored) + f" -> seed {run['seed']}")
    if case in INSTALL:
        cand = CANDIDATES / f"{case}__seed{run['seed']}.pt"
        if cand.exists():
            ckpt = torch.load(cand, map_location="cpu", weights_only=True)
            # "seed" is the training seed of the network; the 80/10/10 split always uses SEED (split_seed)
            ckpt.update(model=MODEL_NAME, loss={"kind": "huber", "beta": run["huber_beta"]}, epochs=run["epochs"],
                        split_seed=SEED, training_run=run["tag"], validation={k: run["val"]["average"][k] for k in METRICS})
            torch.save(ckpt, MODELS / f"{case}.pt")
            print(f"   installed as {(MODELS / f'{case}.pt').relative_to(ROOT)}")
        else:
            ckpt = torch.load(MODELS / f"{case}.pt", map_location="cpu", weights_only=True)
            assert ckpt.get("model") == MODEL_NAME and ckpt.get("seed") == run["seed"], \
                f"{case}: Saved_Models/{case}.pt is not the chosen stage-13 model; run run_offstudy.py --stage 13 first"
    for f in CANDIDATES.glob(f"{case}__seed*.pt"):
        f.unlink()
    for current in ["id", "ig", "is", "ib", "average"]:
        rows.append({"case": case, "model": MODEL_NAME, "current": current,
                     **{k: run["val"][current][k] for k in METRICS}, "train time s": run["seconds"]})

if CANDIDATES.is_dir() and not any(CANDIDATES.iterdir()):
    CANDIDATES.rmdir()
out = ROOT / "ML_Models" / "Results" / "validation_comparison_HUBER.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("written", out.relative_to(ROOT))
