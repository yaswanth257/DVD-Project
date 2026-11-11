"""Train the final ON models (NMOS_ON, PMOS_ON): the Huber MLP recipe plus training rows from the gate operating region.

    python3 ML_Models/Scripts/train_on_models_gate_region.py [--jobs 4]       (after export_huber_models.py; about 7 minutes)

Why: inside a gate an ON transistor has both terminals near one rail and |Vds| of 20 uV - 1 mV, a region uniform sampling of
vd / vs almost never covers; there its gate and body currents (ig, ib) were 2 - 7 % off and set most of the gate-level error.
Data_Generation/Scripts/generate_gate_region_data.py adds 4,000 rows per ON case in that region.

Recipe (as the other final models): MLP 128-64-32-16, SiLU, Huber loss beta = 0.002, AdamW lr 1e-2, 3,000 epochs, the epoch
with the lowest validation loss is kept. Data: the 16,000 training rows of the 80 / 10 / 10 split + 80 % of the gate-region rows
that pass the data-check rule (no zero current, Kirchhoff residual <= 1e-3 of the largest current); validation = the 2,000
validation rows + 10 % of the gate-region rows; the 2,000 test rows and the last 10 % of the gate-region rows are only evaluated.
In gate-region rows with |vd - vs| < 10 mV, id and is are left out of the loss: there the channel current is g x Vds at a tiny Vds,
which the gate composition takes from Kirchhoff's law instead (Gate_Validation/Scripts/gate_common.compose_currents).
Two training seeds per case; the one with the lower sum of the validation median errors (validation rows + gate-region
validation rows) is installed as Saved_Models/<case>.pt. Writes Results/gate_region/runs/<case>__seed<s>.json and
Results/validation_comparison_GATE_REGION.csv. The OFF models are not changed.
"""
import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
CASES, SEEDS = ["NMOS_ON", "PMOS_ON"], [12175305, 12175306]
HIDDEN, BETA, VDS_MASK, EPOCHS = (128, 64, 32, 16), 0.002, 1e-2, 3000
MODEL_NAME = "MLP (Huber loss) + gate-region data"
METRICS = ["R2 (log|I|)", "mean err %", "median err %", "95th pct err %"]
OUT = ROOT / "ML_Models" / "Results" / "gate_region"


def load(case):
    """Main dataset rows and gate-region rows (data-check rule applied), each with its 80 / 10 / 10 split and loss weights."""
    from leakage_model import OUTPUT_COLS, make_features, split_indices
    kind = "nmos" if case.startswith("NMOS") else "pmos"
    df = pd.read_csv(ROOT / "Data_Generation" / "Datasets" / f"{case}_20k.csv")
    gr = pd.read_csv(ROOT / "Data_Generation" / "Datasets" / f"{case}_gate_region.csv")
    I_gr = gr[OUTPUT_COLS].to_numpy()
    keep = ~(I_gr == 0).any(axis=1) & (np.abs(I_gr.sum(axis=1)) / np.abs(I_gr).max(axis=1) <= 1e-3)
    gr = gr[keep].reset_index(drop=True)
    X, X_gr = make_features(df, kind=kind), make_features(gr, kind=kind)
    W_gr = np.ones((len(gr), 4))
    small = (np.abs(gr.vd - gr.vs) < VDS_MASK).to_numpy()
    W_gr[small, 0] = W_gr[small, 2] = 0.0                          # id, is of small-|Vds| rows: not in the loss
    return (kind, X, df[OUTPUT_COLS].to_numpy(), split_indices(len(df)),
            X_gr[list(X.columns)], gr[OUTPUT_COLS].to_numpy(), W_gr, split_indices(len(gr)), int(keep.sum()))


def train(job):
    case, seed = job
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    from leakage_model import OUTPUT_COLS, SEED, build_mlp, error_table
    torch.set_num_threads(1)
    kind, X, I, (itr, iva, ite), X_gr, I_gr, W_gr, (gtr, gva, gte), n_gr = load(case)
    Xtr = np.vstack([X.values[itr], X_gr.values[gtr]]); Itr = np.vstack([I[itr], I_gr[gtr]]); Wtr = np.vstack([np.ones((len(itr), 4)), W_gr[gtr]])
    Xva = np.vstack([X.values[iva], X_gr.values[gva]]); Iva = np.vstack([I[iva], I_gr[gva]]); Wva = np.vstack([np.ones((len(iva), 4)), W_gr[gva]])
    xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(np.log10(np.abs(Itr)))
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, Yt, Wt = T(xs.transform(Xtr)), T(ys.transform(np.log10(np.abs(Itr)))), T(Wtr)
    Xv, Yv, Wv = T(xs.transform(Xva)), T(ys.transform(np.log10(np.abs(Iva)))), T(Wva)

    torch.manual_seed(seed)
    net = build_mlp(list(HIDDEN), Xt.shape[1], 4)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50, min_lr=1e-6)
    huber = nn.SmoothL1Loss(beta=BETA, reduction="none")
    loss_fn = lambda out, y, w: (huber(out, y) * w).sum() / w.sum()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, t0 = float("inf"), None, time.time()
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(Xt), generator=gen).split(256):
            opt.zero_grad()
            loss_fn(net(Xt[b]), Yt[b], Wt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xv), Yv, Wv).item()
        sched.step(val)
        if val < best:
            best, best_state = val, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    seconds = time.time() - t0

    def table(Xr, Ir, W=None):
        with torch.no_grad():
            pred = ys.inverse_transform(net(T(xs.transform(Xr))).numpy())
        t = error_table(np.log10(np.abs(Ir)), pred)
        if W is not None:                                          # gate-region rows: id / is only where they were trained
            err = np.abs(10 ** (pred - np.log10(np.abs(Ir))) - 1) * 100
            for k, c in enumerate(OUTPUT_COLS):
                e = err[W[:, k] > 0, k]
                t.loc[c, ["mean err %", "median err %", "95th pct err %"]] = [e.mean(), np.median(e), np.percentile(e, 95)]
            t.loc["average"] = t.loc[OUTPUT_COLS].mean()
        return {c: t.loc[c, METRICS].to_dict() for c in t.index}

    res = {"case": case, "seed": seed, "model": MODEL_NAME, "hidden": list(HIDDEN), "huber_beta": BETA, "vds_mask_V": VDS_MASK,
           "epochs": EPOCHS, "seconds": seconds, "n_train": len(Itr), "n_gate_region_rows": n_gr,
           "val": table(X.values[iva], I[iva]), "test": table(X.values[ite], I[ite]),
           "gate_region_val": table(X_gr.values[gva], I_gr[gva], W_gr[gva]), "gate_region_test": table(X_gr.values[gte], I_gr[gte], W_gr[gte])}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"{case}__seed{seed}.json").write_text(json.dumps(res, indent=1))
    torch.save({"case": case, "kind": kind, "model": MODEL_NAME, "hidden": list(HIDDEN), "activation": "SiLU",
                "features": list(X.columns), "outputs": OUTPUT_COLS, "x_mean": xs.mean_.tolist(), "x_scale": xs.scale_.tolist(),
                "y_mean": ys.mean_.tolist(), "y_scale": ys.scale_.tolist(), "dataset": f"{case}_20k.csv",
                "extra_training_data": f"{case}_gate_region.csv", "seed": seed, "split": "random", "split_seed": SEED,
                "kirchhoff_lambda": 0.0, "loss": {"kind": "huber", "beta": BETA, "not_in_loss": f"id, is of gate-region rows with |vd - vs| < {VDS_MASK} V"},
                "epochs": EPOCHS, "validation": res["val"]["average"], "state_dict": net.state_dict()},
               OUT / "runs" / f"{case}__seed{seed}.pt")
    a = lambda k: res[k]["average"]
    print(f"{case} seed {seed}: {seconds / 60:.1f} min | val median {a('val')['median err %']:.3f} + gate-region val median "
          f"{a('gate_region_val')['median err %']:.3f} | test median / 95th {a('test')['median err %']:.3f} / {a('test')['95th pct err %']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(train, [(c, s) for c in CASES for s in SEEDS]))
    rows = []
    for case in CASES:
        runs = sorted((r for r in results if r["case"] == case),
                      key=lambda r: r["val"]["average"]["median err %"] + r["gate_region_val"]["average"]["median err %"])
        best = runs[0]
        src = OUT / "runs" / f"{case}__seed{best['seed']}.pt"
        src.replace(ROOT / "ML_Models" / "Saved_Models" / f"{case}.pt")
        for r in runs[1:]:
            (OUT / "runs" / f"{case}__seed{r['seed']}.pt").unlink()
        print(f"{case}: seed {best['seed']} installed as ML_Models/Saved_Models/{case}.pt")
        for current in ["id", "ig", "is", "ib", "average"]:
            rows.append({"case": case, "model": MODEL_NAME, "current": current,
                         **{k: best["val"][current][k] for k in METRICS}, "train time s": best["seconds"]})
    pd.DataFrame(rows).to_csv(ROOT / "ML_Models" / "Results" / "validation_comparison_GATE_REGION.csv", index=False)
    print("written ML_Models/Results/validation_comparison_GATE_REGION.csv")
