"""Shared code for the transistor leakage models.

Used by the evaluation notebook and by Part B / Part C to load a saved model and predict the four terminal
currents of one transistor. The feature code is identical to the training notebooks
(ML_Models/Notebooks/pmos_models.ipynb and nmos_models.ipynb, section 5).

Example
-------
    from leakage_model import LeakageModel
    model = LeakageModel("ML_Models/Saved_Models/PMOS_OFF.pt")
    currents = model.predict(df)          # df has columns vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj
                                          # -> |id|, |ig|, |is|, |ib| in A (magnitudes)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

SEED = 12175305
RAW_FEATURES = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
OUTPUT_COLS = ["id", "ig", "is", "ib"]
K_B, Q_E = 1.380649e-23, 1.602176634e-19


def make_features(df, kind="pmos"):
    """Raw inputs + physics features (10 raw + 15 engineered). kind = "pmos" or "nmos"."""
    f = df[RAW_FEATURES].copy()
    T = df["temp"] + 273.15                     # Kelvin
    vt = K_B * T / Q_E                          # thermal voltage kT/q
    # terminal voltage differences
    f["vgs"] = df["vg"] - df["vs"]
    f["vgd"] = df["vg"] - df["vd"]
    f["vds"] = df["vd"] - df["vs"]
    f["vbs"] = df["vb"] - df["vs"]
    f["vbd"] = df["vb"] - df["vd"]
    # effective source: PMOS -> higher of vd / vs, NMOS -> lower of vd / vs
    if kind == "pmos":
        v_src = np.maximum(df["vd"], df["vs"])
        f["vgs_eff"] = v_src - df["vg"]
    else:
        v_src = np.minimum(df["vd"], df["vs"])
        f["vgs_eff"] = df["vg"] - v_src
    f["vbs_eff"] = df["vb"] - v_src
    f["vds_abs"] = (df["vd"] - df["vs"]).abs()
    # sub-threshold physics
    f["overdrive_vt"] = (f["vgs_eff"] - df["vth0"].abs()) / vt
    f["log_1mexp_vds"] = np.log(np.clip(-np.expm1(-f["vds_abs"] / vt), 1e-12, None))
    f["inv_T"] = 1.0 / T
    f["log_w"] = np.log(df["w"])
    # gate tunnelling ~ oxide field
    f["inv_toxe"] = 1.0 / df["toxe"]
    f["vgs_toxe"] = f["vgs"] / df["toxe"]
    f["vgd_toxe"] = f["vgd"] / df["toxe"]
    return f


def split_indices(n, seed=SEED):
    """80 / 10 / 10 split of row indices, identical to the training notebook."""
    idx = np.arange(n)
    idx_train, idx_tmp = train_test_split(idx, test_size=0.20, random_state=seed)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=seed)
    return idx_train, idx_val, idx_test


def build_mlp(hidden, n_in, n_out):
    """Hidden layers, each followed by SiLU, then a linear output layer."""
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.SiLU()]
        width = h
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def error_table(Y_true_log, Y_pred_log, outputs=OUTPUT_COLS):
    """R2 on log10|I| and the % error |I_pred / I_true - 1| x 100 per current, plus the average over currents."""
    rows = []
    for k, c in enumerate(outputs):
        err = np.abs(10 ** (Y_pred_log[:, k] - Y_true_log[:, k]) - 1) * 100
        rows.append({"current": c, "R2 (log|I|)": r2_score(Y_true_log[:, k], Y_pred_log[:, k]),
                     "mean err %": err.mean(), "median err %": np.median(err),
                     "95th pct err %": np.percentile(err, 95)})
    table = pd.DataFrame(rows).set_index("current")
    table.loc["average"] = table.mean()
    return table


class LeakageModel:
    """A saved transistor model: physics features -> standardization -> MLP -> log10|I| of id, ig, is, ib."""

    def __init__(self, path):
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.info = {k: v for k, v in ckpt.items() if k != "state_dict"}
        self.case, self.kind = ckpt["case"], ckpt["kind"]
        self.features, self.outputs = ckpt["features"], ckpt["outputs"]
        self.x_mean, self.x_scale = np.array(ckpt["x_mean"]), np.array(ckpt["x_scale"])
        self.y_mean, self.y_scale = np.array(ckpt["y_mean"]), np.array(ckpt["y_scale"])
        self.net = build_mlp(ckpt["hidden"], len(self.features), len(self.outputs))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

    def predict_log(self, df):
        """log10 of the current magnitudes, shape (n, 4), columns id, ig, is, ib."""
        X = make_features(df, kind=self.kind)[self.features].values
        Xs = (X - self.x_mean) / self.x_scale
        with torch.no_grad():
            Ys = self.net(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return Ys * self.y_scale + self.y_mean

    def predict(self, df):
        """Current magnitudes |id|, |ig|, |is|, |ib| in A."""
        return pd.DataFrame(10 ** self.predict_log(df), columns=self.outputs, index=df.index)
