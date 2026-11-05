import os
import re
import subprocess

import pandas as pd

from generate_pmos_on_inputs import generate_pmos_on_inputs


MODEL_FILE = "45nm_MGK.pm"
NETLIST_FILE = "Data_Generation/Scripts/PMOS_ON_sample.cir"
OUTPUT_FILE = "Data_Generation/Datasets/PMOS_ON.csv"

LMIN = 45e-9
TEST_SAMPLE_COUNT = None

NGSPICE_COMMAND = "ngspice_con"  # Console executable for automated ngspice simulations


def format_value(value):
    return f"{float(value):.16g}"


def create_netlist(row):

    netlist = f"""* Single PMOS ON

.include "{MODEL_FILE}"

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9

.temp {format_value(row["temp"])}

M1 drain gate source body pmos W={format_value(row["w"])} L={LMIN}

Vd drain 0 DC {format_value(row["vd"])}
Vg gate 0 DC {format_value(row["vg"])}
Vs source 0 DC {format_value(row["vs"])}
Vb body 0 DC {format_value(row["vb"])}

.control
set numdgt=15

altermod pmos vth0 = {format_value(row["vth0"])}
altermod pmos toxe = {format_value(row["toxe"])}
altermod pmos ndep = {format_value(row["ndep"])}
altermod pmos u0 = {format_value(row["u0"])}
altermod pmos xj = {format_value(row["xj"])}

op

let id_val = i(Vd)
let ig_val = i(Vg)
let is_val = i(Vs)
let ib_val = i(Vb)

echo RESULT $&id_val $&ig_val $&is_val $&ib_val

.endc
.end
"""

    with open(NETLIST_FILE, "w", encoding="utf-8") as file:
        file.write(netlist)


def run_single_simulation(row):

    try:
        create_netlist(row)

        result = subprocess.run(
            [NGSPICE_COMMAND, "-b", NETLIST_FILE],
            capture_output=True,
            text=True,
            errors="replace",
            check=False
        )

        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(f"ngspice simulation failed:\n{output}")

        match = re.search(
            r"RESULT\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
            r"([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            output
        )

        if match is None:
            raise RuntimeError(f"Could not extract currents:\n{output}")

        return [float(value) for value in match.groups()]

    finally:
        if os.path.exists(NETLIST_FILE):
            os.remove(NETLIST_FILE)


def run_spice_simulations(df):

    currents = []

    for _, row in df.iterrows():
        currents.append(run_single_simulation(row))

    return pd.DataFrame(currents, columns=["id", "ig", "is", "ib"])


def main():

    df = generate_pmos_on_inputs()

    if TEST_SAMPLE_COUNT is not None:
        df = df.head(TEST_SAMPLE_COUNT).copy()

    print(f"Starting PMOS_ON simulations: {len(df)} samples")

    current_df = run_spice_simulations(df)

    final_df = pd.concat([df.reset_index(drop=True), current_df], axis=1)

    kcl_residual = final_df[["id", "ig", "is", "ib"]].sum(axis=1)

    print("Maximum absolute KCL residual:",
          f"{kcl_residual.abs().max():.6e} A")

    if TEST_SAMPLE_COUNT is None:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("Dataset saved:", OUTPUT_FILE)
    else:
        print("Test run only. CSV not saved.")


if __name__ == "__main__":
    main()

