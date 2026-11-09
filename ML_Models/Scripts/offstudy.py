"""Target-transform and split study for the single-transistor leakage models.

Used by `ML_Models/Scripts/run_offstudy.py` (trains and logs the runs) and `ML_Models/Results/nmos_off_study.ipynb`
(tables and plots). It re-uses the features, the random split and the MLP builder of `leakage_model.py`, so a run with
`target="log"`, `scheme="random"` is the plain MLP of the training notebooks.

Two things can be changed relative to the training notebooks:

* the target
    "log"        log10 |I| of id, ig, is, ib (the training notebooks; the sign is kept outside the model)
    "signedlog"  sign(I) * log10(1 + |I| / I0) for every current whose sign changes inside the training rows
                 (id and is of an OFF transistor); the other currents stay log10 |I|.
                 I0 is a quantile q of |I| over the training rows. The target is smooth through I = 0, and the
                 model predicts the sign as well.
* the split (`SCHEMES`): the random 80/10/10 split of the assignment, or structured splits that hold out
  width bands (W/L), temperature windows, or both.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from leakage_model import (OUTPUT_COLS, SEED, build_mlp, make_features, split_indices)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data_Generation" / "Datasets"
L_MIN = 45e-9                                   # channel length of every transistor (assignment: L = Lmin)
HIDDEN = (128, 64, 32, 16)                      # the architecture the training notebooks chose for NMOS_OFF
MLP_PARAMS = dict(lr=1e-2, weight_decay=1e-4, batch_size=256, max_epochs=3000, patience=300)


# ----------------------------------------------------------------------------------------------------------------
# data
def load_case(case):
    """Dataset of one case with its physics features, signed currents, W/L and temperature."""
    df = pd.read_csv(DATA_DIR / f"{case}_20k.csv")
    kind = "nmos" if case.startswith("NMOS") else "pmos"
    X = make_features(df, kind=kind)
    return df, X, df[OUTPUT_COLS].to_numpy(dtype=np.float64), kind


# ----------------------------------------------------------------------------------------------------------------
# splits
def _band(x, lo, hi):
    return (x >= lo) & (x < hi)


def _bands(x, bands):
    out = np.zeros(len(x), dtype=bool)
    for lo, hi in bands:
        out |= _band(x, lo, hi)
    return out


def make_split(scheme, df):
    """(train, validation, test) index arrays. W/L = w / 45 nm (L is fixed), temperature in degC.

    random      the assignment split: random 80 / 10 / 10 (seed 12175305)
    W_interp    test W/L in [3, 3.5) and [6, 6.5); validation [2, 2.5) and [4.5, 5); all other widths train
    W_extrap    train W/L < 6; validation [6, 7); test >= 7 (wide transistors)
    T_interp    test T in [22, 32) (room temperature) and [80, 90) (85 degC); validation [-10, 0) and [55, 65); rest train
    T_extrap    train T < 85 degC; validation [85, 105); test >= 105 degC (hot)
    WT_blocks   4 x 4 grid of W/L x T blocks: test blocks (W2, T2) and (W3, T3), validation blocks (W2, T3) and (W3, T2),
                the other 12 blocks train - every W band and every T band is seen in training, only the combination is new
    WT_corner   train W/L < 6 and T < 85; validation: the shell just outside (W/L < 7 and T < 105, not train);
                test: wide and hot (W/L >= 7 and T >= 105). Rows outside these three sets are unused.
    """
    n = len(df)
    wl = df["w"].to_numpy() / L_MIN
    T = df["temp"].to_numpy()
    idx = np.arange(n)
    if scheme == "random":
        return split_indices(n)
    if scheme == "W_interp":
        te, va = _bands(wl, [(3, 3.5), (6, 6.5)]), _bands(wl, [(2, 2.5), (4.5, 5)])
    elif scheme == "W_extrap":
        te, va = wl >= 7, _band(wl, 6, 7)
    elif scheme == "T_interp":
        te, va = _bands(T, [(22, 32), (80, 90)]), _bands(T, [(-10, 0), (55, 65)])
    elif scheme == "T_extrap":
        te, va = T >= 105, _band(T, 85, 105)
    elif scheme == "WT_blocks":
        wb = np.digitize(wl, [2.75, 4.5, 6.25])           # 0..3
        tb = np.digitize(T, [1.25, 42.5, 83.75])          # 0..3
        te = ((wb == 1) & (tb == 1)) | ((wb == 2) & (tb == 2))
        va = ((wb == 1) & (tb == 2)) | ((wb == 2) & (tb == 1))
    elif scheme == "WT_corner":
        te = (wl >= 7) & (T >= 105)
        tr = (wl < 6) & (T < 85)
        va = (wl < 7) & (T < 105) & ~tr
        return idx[tr], idx[va], idx[te]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    tr = ~(te | va)
    return idx[tr], idx[va], idx[te]


SCHEMES = ["random", "W_interp", "W_extrap", "T_interp", "T_extrap", "WT_blocks", "WT_corner"]


# ----------------------------------------------------------------------------------------------------------------
# targets
class Target:
    """Encodes signed currents (n, 4) into the model target and back."""

    def __init__(self, kind, I_train, q=0.01):
        assert kind in ("log", "signedlog")
        self.kind, self.q = kind, q
        sign = np.sign(I_train)
        self.changes_sign = [bool((sign[:, k] > 0).any() and (sign[:, k] < 0).any()) for k in range(4)]
        self.fixed_sign = [float(np.sign(I_train[:, k].mean())) for k in range(4)]
        self.signed = [kind == "signedlog" and self.changes_sign[k] for k in range(4)]
        self.i0 = [float(np.quantile(np.abs(I_train[:, k]), q)) if self.signed[k] else 1.0 for k in range(4)]
        self.scaler = None

    def encode(self, I):
        T = np.empty_like(I, dtype=np.float64)
        for k in range(4):
            a = np.abs(I[:, k])
            T[:, k] = np.sign(I[:, k]) * np.log10(1 + a / self.i0[k]) if self.signed[k] else np.log10(a)
        return T

    def fit_scaler(self, T_train):
        self.scaler = StandardScaler().fit(T_train)
        return self.scaler

    def decode(self, T, true_sign=None):
        """Signed currents. `log` targets carry no sign for a current that changes sign: true_sign (n, 4) is used."""
        I = np.empty_like(T, dtype=np.float64)
        for k in range(4):
            if self.signed[k]:
                I[:, k] = np.sign(T[:, k]) * self.i0[k] * (10 ** np.abs(T[:, k]) - 1)
            else:
                sgn = true_sign[:, k] if (self.changes_sign[k] and true_sign is not None) else self.fixed_sign[k]
                I[:, k] = sgn * 10 ** T[:, k]
        return I

    def decode_torch(self, T, S):
        """Same as decode, differentiable, for the Kirchhoff term. S: true signs (batch, 4)."""
        cols = []
        for k in range(4):
            if self.signed[k]:
                cols.append(torch.sign(T[:, k]) * self.i0[k] * (10 ** T[:, k].abs() - 1))
            else:
                cols.append(S[:, k] * 10 ** T[:, k])
        return torch.stack(cols, dim=1)

    def describe(self):
        return {"kind": self.kind, "q": self.q, "signed": self.signed, "i0": self.i0}


# ----------------------------------------------------------------------------------------------------------------
# metrics
def evaluate(I_true, I_pred, floor_frac=0.01):
    """Per-current table (+ average) of a prediction. Arrays are signed currents, shape (n, 4).

    mag err %        | |I_pred| / |I_true| - 1 | x 100        (the metric of the training notebooks; ignores the sign)
    signed err %     | I_pred / I_true - 1 | x 100            (a wrong sign counts: error >= 100 %)
    sign acc %       share of rows with the right sign
    floored err %    | I_pred - I_true | / max(|I_true|, 1 % of the largest current of that row) x 100
    abs err (A)      | I_pred - I_true | (mean, max); sum err % = sum |err| / sum |I_true| x 100
    """
    scale = np.abs(I_true).max(axis=1)
    rows = []
    for k, c in enumerate(OUTPUT_COLS):
        t, p = I_true[:, k], I_pred[:, k]
        mag = np.abs(np.abs(p) / np.abs(t) - 1) * 100
        sgd = np.abs(p / t - 1) * 100
        ab = np.abs(p - t)
        fl = ab / np.maximum(np.abs(t), floor_frac * scale) * 100
        r2 = r2_score(np.log10(np.abs(t)), np.log10(np.clip(np.abs(p), 1e-30, None)))
        rows.append({"current": c, "R2 (log|I|)": r2,
                     "mean err %": mag.mean(), "median err %": np.median(mag), "95th pct err %": np.percentile(mag, 95),
                     "signed mean err %": sgd.mean(), "signed median err %": np.median(sgd),
                     "signed 95th pct err %": np.percentile(sgd, 95),
                     "sign acc %": 100 * np.mean(np.sign(p) == np.sign(t)),
                     "floored mean err %": fl.mean(),
                     "mean abs err (A)": ab.mean(), "max abs err (A)": ab.max(),
                     "sum err %": 100 * ab.sum() / np.abs(t).sum()})
    table = pd.DataFrame(rows).set_index("current")
    table.loc["average"] = table.mean()
    return table


# ----------------------------------------------------------------------------------------------------------------
# training
def train_mlp(Xs_tr, Ts_tr, Xs_va, Ts_va, *, hidden=HIDDEN, seed=SEED, lam=0.0, target=None, S_tr=None,
              loss_kind="mse", huber_beta=0.05, sched_kind="plateau", select="loss",
              lr, weight_decay, batch_size, max_epochs, patience):
    """Same loop as the training notebooks (AdamW, ReduceLROnPlateau, early stopping on the validation MSE).

    lam > 0 adds the Kirchhoff penalty lam * mean[(sum_k I_k / sum_k |I_k|)^2] on the decoded currents.
    Inputs and targets are already standardized; `target` supplies the decode used by the penalty.

    loss_kind   "mse" (notebooks) or "huber" (smooth L1 with transition huber_beta, in standardized units: rows whose
                log error is larger than that, e.g. the near-zero-crossing rows, no longer dominate the fit)
    sched_kind  "plateau" (notebooks: halve the learning rate after 50 epochs without validation improvement) or
                "cosine" (cosine decay from lr to 1e-5 over max_epochs)
    select      which epoch is kept: "loss" = lowest validation loss (notebooks), "median" = lowest validation median
                relative error of the four currents (log targets only; robust to a few outlier rows)
    """
    torch.manual_seed(seed)
    as_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, Yt, Xv, Yv = as_t(Xs_tr), as_t(Ts_tr), as_t(Xs_va), as_t(Ts_va)
    net = build_mlp(list(hidden), Xs_tr.shape[1], 4)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if sched_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50, min_lr=1e-6)
    loss_fn = nn.SmoothL1Loss(beta=huber_beta) if loss_kind == "huber" else nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    if select == "median":
        assert not any(target.signed), "select='median' needs log targets"
        sd_v = as_t(target.scaler.scale_)
    if lam > 0:
        St = as_t(S_tr)
        mu, sd = as_t(target.scaler.mean_), as_t(target.scaler.scale_)

    best, best_state, wait, hist = float("inf"), None, 0, {"train": [], "val": []}
    for epoch in range(max_epochs):
        net.train()
        losses = []
        for b in torch.randperm(len(Xt), generator=gen).split(batch_size):
            opt.zero_grad()
            out = net(Xt[b])
            loss = loss_fn(out, Yt[b])
            if lam > 0:
                I = target.decode_torch(out * sd + mu, St[b])
                loss = loss + lam * ((I.sum(1) / I.abs().sum(1).clamp_min(1e-30)) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            pv = net(Xv)
            val = loss_fn(pv, Yv).item()
            score = val if select == "loss" else ((10 ** ((pv - Yv) * sd_v) - 1).abs().median(dim=0).values.mean().item())
        hist["train"].append(float(np.mean(losses)))
        hist["val"].append(score)
        if sched_kind == "cosine":
            sched.step()
        else:
            sched.step(val)
        if score < best:
            best, best_state, wait = score, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return net, hist


def predict(net, Xs):
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xs, dtype=torch.float32)).numpy()


# ----------------------------------------------------------------------------------------------------------------
def run_tag(case, scheme, target, q, lam, seed, hidden=HIDDEN, patience=MLP_PARAMS["patience"],
            loss_kind="mse", sched_kind="plateau", select="loss", n_train=None, huber_beta=0.05):
    qtag = f"_q{q:g}" if target == "signedlog" else ""
    htag = "" if tuple(hidden) == HIDDEN else "__h" + "-".join(map(str, hidden))
    ptag = "" if patience == MLP_PARAMS["patience"] else f"__pat{patience}"
    vtag = ("" if loss_kind == "mse" else f"__{loss_kind}" + ("" if huber_beta == 0.05 else f"{huber_beta:g}")) + ("" if sched_kind == "plateau" else f"__{sched_kind}") \
        + ("" if select == "loss" else f"__sel{select}") + ("" if n_train is None else f"__n{n_train}")
    return f"{case}__{scheme}__{target}{qtag}__lam{lam:g}__seed{seed}{htag}{ptag}{vtag}"


def run_one(case, scheme="random", target="log", q=0.01, lam=0.0, seed=SEED, hidden=HIDDEN, patience=MLP_PARAMS["patience"],
            loss_kind="mse", sched_kind="plateau", select="loss", n_train=None, huber_beta=0.05, out_dir=None, save_model=None):
    """Train one model and write `<tag>.json` (metrics) and `<tag>.npz` (held-out predictions). Resumable."""
    out_dir = Path(out_dir or ROOT / "ML_Models" / "Results" / "offstudy")
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)
    hidden = tuple(hidden)
    tag = run_tag(case, scheme, target, q, lam, seed, hidden, patience, loss_kind, sched_kind, select, n_train, huber_beta)
    path = out_dir / "runs" / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())

    df, X, I, kind = load_case(case)
    i_tr, i_va, i_te = make_split(scheme, df)
    if n_train is not None:                              # learning curve: a fixed random subset of the training rows
        i_tr = np.sort(np.random.default_rng(SEED).permutation(i_tr)[:n_train])
    tgt = Target(target, I[i_tr], q)
    T = tgt.encode(I)
    xs = StandardScaler().fit(X.values[i_tr])
    ts = tgt.fit_scaler(T[i_tr])
    Xs = lambda i: xs.transform(X.values[i])
    Ts = lambda i: ts.transform(T[i])

    t0 = time.time()
    net, hist = train_mlp(Xs(i_tr), Ts(i_tr), Xs(i_va), Ts(i_va), hidden=hidden, seed=seed, lam=lam, target=tgt,
                          S_tr=np.sign(I[i_tr]), loss_kind=loss_kind, huber_beta=huber_beta, sched_kind=sched_kind, select=select,
                          **{**MLP_PARAMS, "patience": patience})
    seconds = time.time() - t0

    res = {"tag": tag, "case": case, "scheme": scheme, "target": target, "q": q, "lam": lam, "seed": seed, "hidden": list(hidden), "patience": patience, "loss_kind": loss_kind, "huber_beta": huber_beta, "sched_kind": sched_kind, "select": select,
           "n_train": len(i_tr), "n_val": len(i_va), "n_test": len(i_te), "epochs": len(hist["val"]),
           "seconds": seconds, "target_info": tgt.describe()}
    preds = {}
    for part, i in (("train", i_tr), ("val", i_va), ("test", i_te)):
        P = tgt.decode(ts.inverse_transform(predict(net, Xs(i))), true_sign=np.sign(I[i]))
        tab = evaluate(I[i], P)
        res[part] = {c: tab.loc[c].to_dict() for c in tab.index}
        if part != "train":
            preds[part] = P.astype(np.float32)
            preds[f"{part}_true"] = I[i].astype(np.float32)
            preds[f"{part}_idx"] = i
    np.savez_compressed(out_dir / "preds" / f"{tag}.npz", **preds)
    path.write_text(json.dumps(res, indent=1))
    if save_model:
        torch.save({"case": case, "kind": kind, "hidden": list(hidden), "activation": "SiLU",
                    "features": list(X.columns), "outputs": OUTPUT_COLS,
                    "x_mean": xs.mean_.tolist(), "x_scale": xs.scale_.tolist(),
                    "y_mean": ts.mean_.tolist(), "y_scale": ts.scale_.tolist(),
                    "target_transform": tgt.describe(), "fixed_sign": tgt.fixed_sign,
                    "dataset": f"{case}_20k.csv", "seed": seed, "split": scheme, "kirchhoff_lambda": lam,
                    "state_dict": net.state_dict()}, save_model)
    return res
