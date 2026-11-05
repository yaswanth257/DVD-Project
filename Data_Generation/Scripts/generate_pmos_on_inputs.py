
import re

import numpy as np
import pandas as pd


# Project configuration
SEED = 12175305
NUM_SAMPLES = 10000

NOMINAL_VDD = 0.9
LMIN = 45e-9

VDD_MIN = NOMINAL_VDD * 0.9
VDD_MAX = NOMINAL_VDD * 1.1

MODEL_FILE = "45nm_MGK.pm"


def extract_pmos_parameters(model_file):

    with open(model_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Extract only the PMOS model section (from .model pmos to end of file)
    match = re.search(
        r"(?ims)^\s*\.model\s+pmos\b(.*)",
        text
    )

    if match is None:
        raise ValueError("PMOS model section not found.")

    pmos_section = match.group(1)

    parameter_names = ["vth0", "toxe", "ndep", "u0", "xj"]
    nominal = {}

    for name in parameter_names:

        value = re.search(
            rf"\b{name}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            pmos_section,
            re.IGNORECASE
        )

        if value is None:
            raise ValueError(f"Parameter {name} not found.")

        nominal[name] = float(value.group(1))

    return nominal


def generate_pmos_on_inputs():

    rng = np.random.default_rng(SEED)

    nominal = extract_pmos_parameters(MODEL_FILE)

    # Generate VDD first
    vdd = rng.uniform(VDD_MIN, VDD_MAX, NUM_SAMPLES)

    # Generate drain and source voltages independently (PMOS: source near VDD, drain lower)
    vd = rng.uniform(0, vdd)
    vs = rng.uniform(0, vdd)

    # Generate other operating parameters
    temp = rng.uniform(-40, 125, NUM_SAMPLES)

    w = rng.uniform(LMIN, 8 * LMIN, NUM_SAMPLES)

    # Generate process parameters within nominal +/- 10%
    process = {}

    for name, value in nominal.items():

        process[name] = rng.uniform(0.9 * abs(value), 1.1 * abs(value), NUM_SAMPLES)

        # Preserve the sign of the nominal value
        if value < 0:
            process[name] = -process[name]

    # PMOS ON conditions: vg = 0 (VSS), vb = vdd (body tied to VDD)
    vg = np.zeros(NUM_SAMPLES)
    vb = vdd.copy()

    # Construct the dataset in the required column order
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": temp,
        "w": w,
        "vd": vd,
        "vs": vs,
        "vg": vg,
        "vb": vb,
        "vth0": process["vth0"],
        "toxe": process["toxe"],
        "ndep": process["ndep"],
        "u0": process["u0"],
        "xj": process["xj"]
    })

    # Verify the generated dataset
    assert df.shape == (NUM_SAMPLES, 12)
    assert df.isnull().sum().sum() == 0

    assert df["vdd"].between(VDD_MIN, VDD_MAX).all()
    assert df["temp"].between(-40, 125).all()
    assert df["w"].between(LMIN, 8 * LMIN).all()

    assert (df["vd"] >= 0).all()
    assert (df["vd"] <= df["vdd"]).all()

    assert (df["vs"] >= 0).all()
    assert (df["vs"] <= df["vdd"]).all()

    assert (df["vg"] == 0).all()
    assert (df["vb"] == df["vdd"]).all()

    # Verify process parameter ranges (using absolute values for vth0 which is negative)
    for name, value in nominal.items():
        lo = min(0.9 * value, 1.1 * value)
        hi = max(0.9 * value, 1.1 * value)
        assert df[name].between(lo, hi).all()

    return df


if __name__ == "__main__":

    df = generate_pmos_on_inputs()

    print("PMOS_ON input dataset generated successfully!")
    print("Shape:", df.shape)
    print("Random seed:", SEED)

    print("\nNominal PMOS parameters:")

    nominal = extract_pmos_parameters(MODEL_FILE)

    for name, value in nominal.items():
        print(f"{name}: {value}")

    print("\nFirst five samples:")
    print(df.head().to_string(index=False))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset generated in memory. No CSV file saved.")
