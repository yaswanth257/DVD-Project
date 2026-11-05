
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


def extract_nmos_parameters(model_file):

    with open(model_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Extract only the NMOS model section
    match = re.search(
        r"(?ims)^\s*\.model\s+nmos\b(.*?)(?=^\s*\.model\s+pmos\b)",
        text
    )

    if match is None:
        raise ValueError("NMOS model section not found.")

    nmos_section = match.group(1)

    parameter_names = ["vth0", "toxe", "ndep", "u0", "xj"]
    nominal = {}

    for name in parameter_names:

        value = re.search(
            rf"\b{name}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            nmos_section,
            re.IGNORECASE
        )

        if value is None:
            raise ValueError(f"Parameter {name} not found.")

        nominal[name] = float(value.group(1))

    return nominal


def generate_nmos_on_inputs():

    rng = np.random.default_rng(SEED)

    nominal = extract_nmos_parameters(MODEL_FILE)

    # Generate VDD first
    vdd = rng.uniform(VDD_MIN, VDD_MAX, NUM_SAMPLES)

    # Generate drain and source voltages independently
    vd = rng.uniform(0, vdd)
    vs = rng.uniform(0, vdd)

    # Generate other operating parameters
    temp = rng.uniform(-40, 125, NUM_SAMPLES)

    w = rng.uniform(LMIN, 8 * LMIN, NUM_SAMPLES)

    # Generate process parameters within nominal +/- 10%
    process = {}

    for name, value in nominal.items():

        process[name] = rng.uniform(0.9 * value, 1.1 * value, NUM_SAMPLES)

    # NMOS ON conditions
    vg = vdd.copy()
    vb = np.zeros(NUM_SAMPLES)

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

    assert (df["vg"] == df["vdd"]).all()
    assert (df["vb"] == 0).all()

    # Verify process parameter ranges
    for name, value in nominal.items():

        assert df[name].between(0.9 * value, 1.1 * value).all()

    return df


if __name__ == "__main__":

    df = generate_nmos_on_inputs()

    print("NMOS_ON input dataset generated successfully!")
    print("Shape:", df.shape)
    print("Random seed:", SEED)

    print("\nNominal NMOS parameters:")

    nominal = extract_nmos_parameters(MODEL_FILE)

    for name, value in nominal.items():
        print(f"{name}: {value}")

    print("\nFirst five samples:")
    print(df.head().to_string(index=False))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset generated in memory. No CSV file saved.")
