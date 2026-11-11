"""Torch-free loader for the saved transistor models (fallback when PyTorch is not installed).

Reads ML_Models/Saved_Models/*.pt directly (a zip + pickle) and runs the MLP forward pass (Linear -> SiLU ... -> Linear) in
numpy, using the same make_features / apply_signs code as ML_Models/Scripts/leakage_model.py. Same predictions up to float32
rounding; `LeakageModelNP` has the same predict / predict_signed interface as `LeakageModel`.
"""
import pickle
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# leakage_model.py imports torch at the top; give it a temporary stub so its pure-numpy helpers can be imported without torch
import sklearn.metrics, sklearn.model_selection  # noqa: E401,F401  (import first so scipy/sklearn never see the stub)
_stubbed = False
try:
    import torch  # noqa: F401
except ImportError:
    _t = types.ModuleType("torch"); _t.nn = types.SimpleNamespace(Module=object, Sequential=object, Linear=object, SiLU=object)
    sys.modules["torch"] = _t; sys.modules["torch.nn"] = _t.nn; _stubbed = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ML_Models" / "Scripts"))
from leakage_model import apply_signs, make_features  # noqa: E402
if _stubbed:
    sys.modules.pop("torch", None); sys.modules.pop("torch.nn", None)


class _Storage:
    def __init__(self, dtype): self.dtype = dtype


def _rebuild(storage, offset, size, stride, *args):
    return ("tensor", storage, offset, tuple(size), tuple(stride))


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f); self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2": return _rebuild
        if name == "OrderedDict": from collections import OrderedDict; return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}[name]
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _, dtype, key, _loc, _n = pid
        return np.frombuffer(self.zf.read(f"{self.prefix}/data/{key}"), dtype=dtype)


def load_checkpoint(path):
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split("/")[0]
        with zf.open(f"{prefix}/data.pkl") as f:
            ckpt = _Unpickler(f, zf, prefix).load()
    sd = {}
    for k, (_, storage, off, size, stride) in ckpt["state_dict"].items():
        n = int(np.prod(size)) if size else 1
        sd[k] = np.array(storage[off:off + n], dtype=np.float64).reshape(size)
    ckpt["state_dict"] = sd
    return ckpt


class LeakageModelNP:
    def __init__(self, path):
        ckpt = load_checkpoint(path)
        self.info = {k: v for k, v in ckpt.items() if k != "state_dict"}
        self.case, self.kind = ckpt["case"], ckpt["kind"]
        self.features, self.outputs = ckpt["features"], ckpt["outputs"]
        self.x_mean, self.x_scale = np.array(ckpt["x_mean"]), np.array(ckpt["x_scale"])
        self.y_mean, self.y_scale = np.array(ckpt["y_mean"]), np.array(ckpt["y_scale"])
        sd = ckpt["state_dict"]
        n_lin = len([k for k in sd if k.endswith(".weight")])
        self.layers = [(sd[f"{2 * i}.weight"], sd[f"{2 * i}.bias"]) for i in range(n_lin)]   # Linear at even indices

    def predict_log(self, df):
        X = make_features(df, kind=self.kind)[self.features].values
        h = (X - self.x_mean) / self.x_scale
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = h / (1.0 + np.exp(-h))                      # SiLU
        return h * self.y_scale + self.y_mean

    def predict(self, df):
        return pd.DataFrame(10 ** self.predict_log(df), columns=self.outputs, index=df.index)

    def predict_signed(self, df):
        signed = apply_signs(self.case, self.predict(df).to_numpy(), df["vd"].to_numpy(), df["vs"].to_numpy())
        return pd.DataFrame(signed, columns=self.outputs, index=df.index)


def load_model(path):
    """Real torch model if torch is installed, otherwise the numpy re-implementation."""
    try:
        import torch as _torch
        if hasattr(_torch, "load"):
            from leakage_model import LeakageModel
            return LeakageModel(path)
    except ImportError:
        pass
    return LeakageModelNP(path)
