"""Step 9 - ML prediction for the CLA: sum gate leakages from Part B gate_ml_results.csv.

For each of 128 input combinations x 200 corners the CLA ML leakage is the sum of the
ml_leak_total values of all 33 gate instances in the circuit. Each gate instance uses the
(gate_type, input_combo) that the actual logic values on its inputs produce at that input
combination of the full CLA.

This approach is correct under the assignment assumption that every gate input sits at
exactly 0 V or VDD (the same assumption used in Part B). No new SPICE runs are needed.

Usage (from repository root):
    python3 Test_Circuit/Scripts/09_predict_ml_cla.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Gate_Validation" / "Scripts"))

import numpy as np
import pandas as pd
from circuit_common import (GATE_ML_CSV, GATES, ML_CSV, all_input_combos, combo_label,
                             evaluate_circuit)
from gate_common import combo_label as gate_combo_label


if __name__ == "__main__":
    # Load Part B gate ML results
    gml = pd.read_csv(GATE_ML_CSV, dtype={"input": str})          # keep labels such as "01" as text

    # Fast lookup: (gate_type, input label, corner) -> ml_leak_total
    lookup = {(row.gate, row.input, int(row.corner)): row.ml_leak_total for row in gml.itertuples(index=False)}

    combos = all_input_combos()
    corners_df = gml[["corner", "vdd", "temp"]].drop_duplicates().sort_values("corner")
    corners = corners_df.to_dict("records")

    rows = []
    for c in corners:
        corner_id = int(c["corner"])
        for combo in combos:
            # Evaluate the full CLA logic for this input combination
            nets = evaluate_circuit(*combo)
            ml_total = 0.0
            for (inst, gate_type, in_nets, out_net) in GATES:
                # Determine the logic bits at this gate's inputs
                in_bits = tuple(nets[n] for n in in_nets)
                gate_input_label = gate_combo_label(in_bits)          # e.g. (0, 1) -> "01", as in gate_ml_results
                key = (gate_type, gate_input_label, corner_id)
                if key not in lookup:
                    raise KeyError(f"No entry in gate_ml_results for {key}")
                ml_total += lookup[key]
            rows.append({
                "corner":        corner_id,
                "vdd":           c["vdd"],
                "temp":          c["temp"],
                "input_combo":   combo_label(combo),
                "ml_leak_total": ml_total,
            })

    df = pd.DataFrame(rows)
    ML_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ML_CSV, index=False)
    print(f"{len(df)} ML predictions -> {ML_CSV}")
    print("Median ml_leak_total:", df["ml_leak_total"].median())
